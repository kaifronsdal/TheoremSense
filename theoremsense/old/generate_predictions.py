# Description: Generate predictions for a dataset using a trained model.
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # "2,3,6,7"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import argparse
from dataset import from_name
from vllm import LLM, SamplingParams
from transformers import pipeline

from prompt import generate_nshot_prompts
from util import save_output
import torch

from pathlib import Path
from tqdm import tqdm
import contextlib
from enum import Enum


@contextlib.contextmanager
def suppress_print():
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


class Method(Enum):
    AUTOREGRESSIVE = 'autoregressive'
    TEACHER_FORCING = 'teacher_forcing'


METHOD_MAP = {
    'ar': Method.AUTOREGRESSIVE,
    'autoregressive': Method.AUTOREGRESSIVE,
    'tf': Method.TEACHER_FORCING,
    'teacher_forcing': Method.TEACHER_FORCING
}


def preprocess_dataset(dataset, split, nshot=True, n=3):
    if split == 'train':
        dataset = dataset['train']
    elif split == 'test':
        dataset = dataset['test']
    elif split == 'all':
        # dataset = dataset.all
        raise NotImplementedError('Split all not implemented yet.')
    else:
        raise ValueError(f'Invalid split {split}.')

    if nshot:
        return generate_nshot_prompts(dataset, n)
    else:
        return dataset


@save_output
def generate_autoregressive(prompts, solutions, llm, sampling_params):
    # chunk_size = generate_config.get('batch_size', 8) * 4

    outputs = llm.generate(prompts, sampling_params)

    # outputs = []
    # for i in tqdm(range(0, len(prompts), chunk_size)):
    #     batch_prompts = prompts[i:i + chunk_size]
    #     batch_outputs = pipe(batch_prompts, **generate_config)
    #     outputs.extend(batch_outputs)

    return outputs, solutions


def batched_teacher_forcing_predictions(prompts, solutions, model, tokenizer, device, debug=False):
    input = tokenizer([f"{p}\n\n{s}" for p, s in zip(prompts, solutions)], return_tensors="pt", padding=True,
                      truncation=True)
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        response = model(input.input_ids.to(device), attention_mask=input.attention_mask.to(device))
    # get the response ids for the solution part of the input
    all_preds = torch.argmax(response.logits, dim=-1).cpu()
    all_logits = response.logits.cpu()

    tf_ids = []
    solution_ids = []
    losses = []
    for i in range(len(prompts)):
        # assumes attention mask is contiguous
        prompt_length = prompt_tokens.attention_mask[i].sum().item()
        solution_start = torch.nonzero(input.attention_mask[i, :] == 1, as_tuple=False)[0, 0].item() + prompt_length
        solution_length = input.attention_mask[i].sum().item() - prompt_length

        # if prompt_tokens.input_ids[i, -1] != input.input_ids[i, solution_start - 1]:
        #     print(prompt_tokens.input_ids[i, -1], input.input_ids[i, solution_start - 1])
        #     breakpoint()
        assert prompt_tokens.input_ids[i, -1] == input.input_ids[i, solution_start - 1]

        # print(input.attention_mask[i])
        # print(f'prompt_tokens[i]: {[tokenizer.decode([t]) for t in prompt_tokens.input_ids[i]]}')
        # print(f'input[i]: {[tokenizer.decode([t]) for t in input.input_ids[i]]}')
        # print(f'all_preds[i]: {[tokenizer.decode([t]) for t in all_preds[i]]}')
        # print(f'solution[i]: {[tokenizer.decode([t]) for t in input.input_ids[i, solution_start:solution_start + solution_length]]}')
        # print(f'predicted: {[tokenizer.decode([t]) for t in all_preds[i, solution_start - 1:solution_start - 1 + solution_length]]}')

        tf_ids.append(all_preds[i, solution_start - 1:solution_start - 1 + solution_length])
        solution_ids.append(input.input_ids[i, solution_start:solution_start + solution_length])
        loss = torch.nn.functional.cross_entropy(all_logits[i, solution_start - 1:solution_start - 1 + solution_length],
                                                 input.input_ids[i, solution_start:solution_start + solution_length])
        losses.append(loss.item())

    if debug:
        for i in range(len(tf_ids)):
            print(f"Prompt: {prompts[i]}")
            assert len(tf_ids[i]) == len(solution_ids[i])
            for j in range(len(tf_ids[i])):
                print(f"Predicted:\t{tokenizer.decode([tf_ids[i][j]], skip_special_tokens=True).__repr__()}")
                print(f"Actual:   \t{tokenizer.decode(solution_ids[i][j], skip_special_tokens=True).__repr__()}")

    return tf_ids, solution_ids, losses


@save_output
def generate_teacher_forcing(prompts, solutions, model, tokenizer,
                             batch_size=8, device=None, debug=False):
    if device is None:
        device = model.device
    teacher_forced_predictions = []
    solution_ids = []
    losses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_solutions = solutions[i:i + batch_size]
        tf_ids, sol_ids, loss = batched_teacher_forcing_predictions(batch_prompts, batch_solutions, model,
                                                                    tokenizer, device, debug)
        teacher_forced_predictions.extend(tf_ids)
        solution_ids.extend(sol_ids)
        losses.extend(loss)

    return teacher_forced_predictions, solution_ids, losses


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for a dataset.')
    parser.add_argument('--model', '-m', type=str, help='Model name.')

    # Dataset parameters
    parser.add_argument('--dataset', '-d', type=str, help='Dataset name(s).')
    parser.add_argument('--subset', type=str, default=None, help='Subset of the dataset to use.')
    parser.add_argument('--split', type=str, default='train', help='Which split of dataset to use (train, test, all)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generating predictions.')

    # Saving parameters
    parser.add_argument('--output', '-o', type=str, help='Path to the output file.')
    parser.add_argument('--override', type=bool, default=False,
                        help='Whether to override the output file if it exists.')
    parser.add_argument('--load', action='store_true',
                        help='Whether to load predictions from file if it exists.')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling.')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate.')
    parser.add_argument('--num_shots', type=int, default=0, help='Number of shots to generate')
    parser.add_argument('--num_logprob', type=int, default=100, help='Number of logprobs to generate')

    # generation method
    parser.add_argument('--method', type=str, default='autoregressive',
                        help='How to calculate the predictions (autoregressive, teacher_forcing)')

    args = parser.parse_args()
    args.method = METHOD_MAP[args.method]

    print('=' * 80)
    print(args)
    print('=' * 80)

    if args.method == Method.TEACHER_FORCING:
        model = pipeline('text-generation', model=args.model, tokenizer=args.model, device_map="auto",
                         torch_dtype=torch.bfloat16, batch_size=args.batch_size)
        tokenizer = model.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_token_id
    elif args.method == Method.AUTOREGRESSIVE:
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                         logprobs=args.num_logprob)

        model = LLM(model=args.model, trust_remote_code=True)
    else:
        raise ValueError(f'Invalid method {args.method}')

    datasets = args.dataset.split(',')
    output = args.output

    with suppress_print():
        datas = from_name(datasets, subset=args.subset)

    for data in datas:
        # clear gpu memory
        torch.cuda.empty_cache()

        dataset_name = data['name']
        data = data['data']

        data = preprocess_dataset(data, split=args.split, nshot=args.num_shots > 0, n=args.num_shots)
        prompts = [d['question'] for d in data]
        solutions = [d['answer'] for d in data]

        save_dir = Path(output).expanduser() / dataset_name.replace("/", "_")
        save_dir.mkdir(parents=True, exist_ok=True)

        if args.method == Method.AUTOREGRESSIVE:
            save_path = save_dir / f'{args.model.replace("/", "_")}_autoregressive.pkl'
            predictions = generate_autoregressive(prompts, solutions, model, sampling_params, output_path=save_path,
                                                  override=args.override)
        elif args.method == Method.TEACHER_FORCING:
            save_path = save_dir / f'{args.model.replace("/", "_")}_teacher_forcing.pkl'
            predictions = generate_teacher_forcing(prompts, solutions, model.model, model.tokenizer,
                                                   output_path=save_path, override=args.override)


if __name__ == "__main__":
    main()
