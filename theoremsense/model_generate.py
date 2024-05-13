import os
from collections import defaultdict

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # "2,3,6,7"
    print('CUDA_VISIBLE_DEVICES not set. Setting to 6')

import argparse
import json
import contextlib
from typing import Literal
from pathlib import Path
import numpy as np
import datasets as hf_datasets

from utils import save_arguments
import torch

from blocks import vLLM, Gemini, HFModel
from blocks import Generator, Prompt, Batch, Block, Map, Retry

# load the name maps
with open('name_maps.json', 'r') as f:
    name_maps = json.load(f)
    DATASET_MAP = name_maps['DATASET_MAP']
    MODEL_MAP = name_maps['MODEL_MAP']


@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


GenerationMethod = Literal['autoregressive', 'teacher_forcing']

METHOD_MAP = {
    'ar': 'autoregressive',
    'autoregressive': 'autoregressive',
    'tf': 'teacher forcing',
    'teacher_forcing': 'teacher_forcing',
    'teacher forcing': 'teacher_forcing'
}

DATASET_CONFIGS = {
    'EleutherAI/hendrycks_math': {
        'subsets': ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                    'number_theory', 'prealgebra', 'precalculus'],
        'aliases': ['hendrycks_math', 'math', 'hendrycks'],
    }
}

DATASET_ALIASES = {alias: dataset_name
                   for dataset_name, config in DATASET_CONFIGS.items()
                   for alias in config['aliases']}
DATASET_ALIASES.update({dataset_name: dataset_name for dataset_name in DATASET_CONFIGS})


class TeacherForcing(Block):
    parallel = True

    def __init__(self, model):
        super().__init__()
        assert isinstance(model, HFModel)
        self.model = model

    def compute_metrics(self, outputs: list[dict]):
        metrics = []
        for output in outputs:
            metric = {}
            # TF accuracy
            # metrics['tfa'].append(np.mean(output['teacher_forced_ids'] == output['solution_ids']))
            metric['tfa'].append(torch.mean((output['teacher_forced_ids'] == output['solution_ids']).float()).item())
            # TFCE
            TFCE = torch.nn.functional.cross_entropy(output['teacher_forced_logits'], output['solution_ids']).item()
            metric['tfce'].append(TFCE)
            # Perplexity
            metric['perpelexity'].append(np.exp(TFCE))
            # sumCE
            CE = torch.nn.functional.cross_entropy(output['teacher_forced_logits'], output['solution_ids'],
                                                   reduction='sum').item()
            metric['sumCE'].append(CE)
            # BPC
            metric['bpc'].append(CE / (output['solution_num_chars'] * np.log(2)))
            # num tokens/chars
            metric['total_num_tokens'].append(output['total_num_tokens'])
            metric['prompt_num_tokens'].append(output['prompt_num_tokens'])
            metric['solution_num_tokens'].append(output['solution_num_tokens'])
            metric['total_num_chars'].append(output['total_num_chars'])
            metric['prompt_num_chars'].append(output['prompt_num_chars'])
            metric['solution_num_chars'].append(output['solution_num_chars'])

            metrics.append(metric)

        return metrics

    def process(self, input):
        if not isinstance(input, Batch):
            input = Batch([input])
        output = self.model.generate_teacher_forcing(input)
        output = self.compute_metrics(output)
        return Batch(output) if isinstance(input, Batch) else output[0]

    def convert_tokens(self, tokens):
        return [self.model.tokenizer.decode(t) for t in tokens]


def get_args():
    parser = argparse.ArgumentParser(description='Generate predictions for a dataset.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling.')

    parser.add_argument('--model', '-m', type=str, help='Model name.')
    parser.add_argument('--backend', type=str, default='vllm', help='Which backend to use (hf, vllm, gemini)')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Size of the tensor parallelism group.')

    # Dataset parameters
    parser.add_argument('--dataset', '-d', type=str, help='Dataset name(s).')
    parser.add_argument('--subset', type=str, default=None, help='Subset of the dataset to use.')
    parser.add_argument('--split', type=str, default='train', help='Which split of dataset to use (train, test, all)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generating predictions.')
    parser.add_argument('--use_chat', action='store_true', help='Whether to use chat template format.')
    parser.add_argument('--detect_chat', action='store_true', help='Whether to detect chat template format.')

    # Saving parameters
    parser.add_argument('--output', '-o', type=str, help='Path to the output folder.')
    parser.add_argument('--override', action='store_true',
                        help='Whether to override the output file if it exists.')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling.')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate.')
    parser.add_argument('--num_shots', type=int, default=0, help='Number of shots to generate')
    # parser.add_argument('--num_logprob', type=int, default=100, help='Number of logprobs to generate')

    # generation method
    parser.add_argument('--method', type=str, default='autoregressive',
                        help='How to calculate the predictions (autoregressive, teacher_forcing)')

    args = parser.parse_args()
    args.method = METHOD_MAP[args.method]

    if args.method == 'teacher_forcing':
        assert args.backend == 'hf', 'Teacher forcing is only supported for HF models.'

    if args.detect_chat:
        assert args.use_chat is False, 'Cannot use both detect_chat and use_chat'
        CHAT_INDICATORS = ['it', 'instruct', 'chat']
        if any(indicator in args.model.lower() for indicator in CHAT_INDICATORS):
            args.use_chat = True
            print('Detected chat template format. Using chat template format.')

    # check if output file exists
    # if Path(args.output).exists() and not args.override:
    #     raise FileExistsError(f'Output folder {args.output} already exists. Use --override to overwrite.')

    return args


def load_model(args):
    terminators = None
    if 'Meta-Llama-3' in args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    if args.backend == 'vllm':
        model = vLLM(model_name=args.model, use_subprocess=False, temperature=args.temperature, top_p=args.top_p,
                     max_tokens=args.max_tokens,
                     tensor_parallel_size=args.tensor_parallel_size,
                     stop_token_ids=terminators)  # , device=args.device)  # , n=args.num_shots)
    elif args.backend == 'gemini':
        model = Gemini(model_name=args.model, temperature=args.temperature, top_p=args.top_p,
                       max_output_tokens=args.max_tokens)  # , candidate_count=args.num_shots)
    elif args.backend == 'hf':
        # TODO: FIX PARAMS
        model = HFModel(model_name=args.model, batch_size=args.batch_size, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_tokens, eos_token_id=terminators)
    else:
        raise ValueError(f'Unknown backend {args.backend}')

    return model


def load_datasets(args):
    datasets = {}
    for dataset_name in args.dataset.split(','):
        dataset_name = dataset_name.strip()
        assert dataset_name in DATASET_ALIASES, f'Unknown dataset {dataset_name}'
        dataset_name = DATASET_ALIASES[dataset_name]

        for subset in DATASET_CONFIGS[dataset_name]['subsets']:
            with suppress_print():
                try:
                    dataset = hf_datasets.load_dataset(dataset_name, subset)[args.split]
                except ValueError as e:  # @TODO: get right exception type
                    print(e)
                    print(f'Error while loading {dataset_name}:{subset}. Redownloading...')
                    dataset = hf_datasets.load_dataset(dataset_name, subset, download_mode='force_redownload',
                                                       verification_mode='no_checks')[args.split]
            datasets[f'{dataset_name}:{subset}'] = dataset

    return datasets


class NShotSample(Block):
    """
    Randomly samples n-shot examples from the batch.
    """
    parallel = True

    def __init__(self, num_shots, subset=None, filter=None, seed=None):
        super().__init__()
        self.num_shots = num_shots

        self.subset = subset
        self.filter = filter
        self.seed = seed

    def process(self, input):
        if not isinstance(input, Batch):
            raise ValueError('NShotSample block must be used with a Batch input.')

        subset = self.subset or list(range(len(input)))

        assert len(input) > self.num_shots, \
            f'Batch size must be at least {self.num_shots + 1} for {self.num_shots}-shot sampling.'
        assert len(subset) > self.num_shots, \
            f'Subset size must be at least {self.num_shots + 1} for {self.num_shots}-shot sampling.'
        assert all(0 <= i < len(input) for i in subset), 'Invalid subset indices.'

        if self.filter:
            subset = [i for i in subset if self.filter(input[i], [input[j] for j in subset])]
            assert len(subset) > self.num_shots, \
                (f'Not enough examples in the subset after filtering. Subset size must be at least '
                 f'{self.num_shots + 1} for {self.num_shots}-shot sampling.')

        # select the few-shot examples
        if self.seed is not None:
            np.random.seed(self.seed)
        outputs = []
        for i, ex in enumerate(input):
            few_shot_sample = np.random.choice(subset, self.num_shots + 1, replace=False)
            # remove the current example from the few-shot sample if it is in the sample
            few_shot_sample = few_shot_sample[few_shot_sample != i][:self.num_shots]

            output = {
                'input': ex,
                'n_shot': [input[i] for i in few_shot_sample]
            }
            outputs.append(output)

        return Batch(outputs)


COT_PROMPT = '\nPlease reason step by step, and put your final answer within \\boxed{}.'

PROMPT = r"""Problem:
Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct."""


# return PROMPT + "\n\n" + "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"

def craft_prompt(input):
    n = len(input['n_shot'])
    _question_format = 'Question {i}: {question}{extra}\n\nAnswer {i}: {answer}'

    prompt = f'Answer the following {n + 1} questions:\n\n'
    for i, sample in enumerate(input['n_shot']):
        prompt += _question_format.format(i=i + 1, question=sample['problem'],
                                          answer=sample['solution'], extra=COT_PROMPT)
        prompt += '\n\n'

    prompt += _question_format.format(i=n + 1, question=input['input']['problem'], answer='', extra=COT_PROMPT)
    return prompt


def craft_chat_prompt(input):
    prompt = []
    for sample in input['n_shot']:
        prompt.append({'role': 'user', 'content': sample['problem'] + COT_PROMPT})
        prompt.append({'role': 'assistant', 'content': sample['solution']})

    prompt.append({'role': 'user', 'content': input['input']['problem'] + COT_PROMPT})
    return prompt


def prompt_generator(args):
    craft_prompt_fn = craft_chat_prompt if args.use_chat else craft_prompt

    def filter_longest(ex, inputs):
        # filter out the longest 50% of the examples
        median = np.percentile([len(e['problem'] + e['solution']) for e in inputs], 50)
        return len(ex['problem'] + ex['solution']) < median

    return NShotSample(args.num_shots, filter=filter_longest, seed=args.seed) >> Map(craft_prompt_fn)


def hard_coded_prompt_generator(args):
    if args.method == 'teacher_forcing':
        def hard_coded_prompt(input):
            return {
                'question': PROMPT + "\n\n" + "Problem:" + "\n" + input['problem'] + "\n\n" + "Solution:",
                'solution': input['solution']
            }

        return Map(hard_coded_prompt)
    else:
        def hard_coded_prompt(input):
            return PROMPT + "\n\n" + "Problem:" + "\n" + input['problem'] + "\n\n" + "Solution:"

        return Map(hard_coded_prompt)


def evaluate(gen, batch, args):
    # if args.method == 'teacher_forcing':
    #     pipeline = prompt_generator(args) >> gen
    # else:
    #     def filter_answer(x: str):
    #         return '\\boxed' in x  # or 'The answer is' in x
    #
    #     pipeline = prompt_generator(args) >> Retry(gen, max_retries=10, filter=filter_answer)
    pipeline = hard_coded_prompt_generator(args) >> gen
    return pipeline(batch)


def eval(args):
    # Load the model
    print(f'Loading model {args.model}...')
    model = load_model(args)
    if args.method == 'teacher_forcing':
        gen = TeacherForcing(model)
    elif args.method == 'autoregressive':
        gen = Generator(model, prepend_history=False)
    else:
        raise ValueError(f'Unknown method {args.method}')

    # Load the datasets
    print(f'Loading dataset {args.dataset}...')
    datasets = load_datasets(args)

    # Generate the predictions
    for dataset_name, dataset in datasets.items():
        print(f'Generating predictions for {dataset_name}...')
        batch = Batch(dataset)

        save_dir = (
                Path(args.output).expanduser()
                / f'{DATASET_MAP[dataset_name]}_{args.split}'
                / MODEL_MAP[args.model]
                / args.method
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_dir.exists() and not args.override:
            print(f'Output path {save_dir} already exists. Use --override to overwrite. Skipping...')
            continue

        predictions = evaluate(gen, batch, args)

        with open(save_dir / 'output.json', 'w') as f:
            json.dump(predictions, f, indent=4)

        save_arguments(args, save_dir / 'args.json')


def debug(args):
    # test load datasets
    print('Testing load_datasets')
    dataset_name = 'EleutherAI/hendrycks_math'
    args.dataset = dataset_name
    datasets = load_datasets(args)
    assert len(datasets) == 7, f'Expected 7 datasets, got {len(datasets)}'
    assert all(f'{dataset_name}:{subset}' in datasets for subset in DATASET_CONFIGS[dataset_name]['subsets']), \
        'Missing dataset subsets'
    assert all(len(dataset) > 0 for dataset in datasets.values()), 'Empty datasets'

    for dataset_name, dataset in datasets.items():
        for ex in dataset:
            assert set(ex.keys()) == {'problem', 'solution', 'level', 'type'}, \
                f'Unexpected keys in example: {ex.keys()}'

    print('load_datasets passed')

    # test nshotblock
    print('Testing NShotSample')
    nshot_block = NShotSample(3)
    batch = Batch([{'problem': f'Question {i}', 'solution': f'Answer {i}'} for i in range(10)])
    output = nshot_block(batch)
    assert len(output) == len(batch), 'Batch size should not change'
    assert all(len(ex['n_shot']) == 3 for ex in output), 'Expected 3 n-shot examples'
    assert all(len({e['problem'] for e in ex['n_shot']}) == 3 for ex in output), 'Expected unique n-shot examples'
    assert all(ex['input'] not in ex['n_shot'] for ex in output), 'Input should not be in n-shot examples'
    assert all(ex['input'] in batch for ex in output), 'Input should be in the original batch'

    print('NShotSample passed')

    # test craft prompt
    print('Testing craft_prompt')
    input = {
        'n_shot': [
            {'problem': 'Question 1', 'solution': 'Answer 1'},
            {'problem': 'Question 2', 'solution': 'Answer 2'},
            {'problem': 'Question 3', 'solution': 'Answer 3'}
        ],
        'input': {'problem': 'Question 4', 'solution': 'Answer 4'}
    }
    prompt = craft_prompt(input)
    from textwrap import dedent
    assert prompt == dedent("""\
    Answer the following 4 questions:

    Question 1: Question 1
    Answer: Answer 1

    Question 2: Question 2
    Answer: Answer 2

    Question 3: Question 3
    Answer: Answer 3

    Question 4: Question 4
    Answer: """), 'Prompt not as expected'

    # test craft chat prompt
    chat_prompt = craft_chat_prompt(input)
    assert chat_prompt == [
        {'role': 'user', 'content': 'Question 1'},
        {'role': 'assistant', 'content': 'Answer 1'},
        {'role': 'user', 'content': 'Question 2'},
        {'role': 'assistant', 'content': 'Answer 2'},
        {'role': 'user', 'content': 'Question 3'},
        {'role': 'assistant', 'content': 'Answer 3'},
        {'role': 'user', 'content': 'Question 4'}
    ], 'Chat prompt not as expected'

    print('craft_prompt passed')


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        debug(args)
    else:
        eval(args)
