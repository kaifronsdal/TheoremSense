"""
Takes in model and dataset, and generates predictions for the dataset. Saves the predictions to a file.
"""

# python theoremsense/generate_predictions.py --model=deepseek-ai/deepseek-math-7b-instruct --dataset=EleutherAI/hendrycks_math --subset=algbra --output=~/generated_outputs --perplexity=True --num_shots=3

import argparse
from dataset import from_name
from vllm import LLM, SamplingParams

from prompt import generate_nshot_prompts
from grader import ExactMatchGrader
from dataset import get_boxed_answer
from latex_formater import latex_deformat

from pathlib import Path
from metrics import perplexity
import numpy as np
import pickle
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


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


def generate_predictions(llm, sampling_params, dataset, output_path, override=False):
    if Path(output_path).exists() and not override:
        raise ValueError(f'Output file {output_path} already exists. Use --override to overwrite it.')

    outputs = llm.generate(dataset, sampling_params)

    with open(output_path, 'wb') as f:
        pickle.dump(outputs, f)

    return outputs


def compute_perplexity(outputs, dataset, save_path, override=False):
    if Path(save_path).exists() and not override:
        raise ValueError(f'Output file {save_path} already exists. Use --override to overwrite it.')

    predictions = [o.outputs[0] for o in outputs]
    ppl = perplexity(predictions)

    print(f'Perplexity for {dataset} is {np.mean(ppl)}.')

    with open(save_path, 'wb') as f:
        pickle.dump(ppl, f)
    return ppl


def grade_predictions(outputs, data, dataset, save_path, override=False):
    if Path(save_path).exists() and not override:
        raise ValueError(f'Output file {save_path} already exists. Use --override to overwrite it.')

    grader = ExactMatchGrader()

    boxed_predictions = [latex_deformat(get_boxed_answer(o.outputs[0].text)) for o in outputs]
    boxed_answers = [latex_deformat(d['boxed']) for d in data]
    grades = grader.grade(boxed_predictions, boxed_answers)

    print(f'Total accuracy for {dataset} is {np.mean(grades)}.')

    with open(save_path, 'wb') as f:
        pickle.dump(grades, f)
    return grades


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for a dataset.')
    parser.add_argument('--model', '-m', type=str, help='Model name.')

    # Dataset parameters
    parser.add_argument('--dataset', '-d', type=str, help='Dataset name(s).')
    parser.add_argument('--subset', type=str, default=None, help='Subset of the dataset to use.')
    parser.add_argument('--split', type=str, default='train', help='Which split of dataset to use (train, test, all)')

    # Saving parameters
    parser.add_argument('--output', '-o', type=str, help='Path to the output file.')
    parser.add_argument('--override', action='store_true',
                        help='Whether to override the output file if it exists.')
    parser.add_argument('--load', action='store_true',
                        help='Whether to load predictions from file if it exists.')

    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling.')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate.')
    parser.add_argument('--num_logprob', type=int, default=100, help='Number of logprob to output')
    parser.add_argument('--num_shots', type=int, default=0, help='Number of shots to generate')

    # post processing parameters
    parser.add_argument('--perplexity', action='store_true', help='Whether to calculate perplexity')
    parser.add_argument('--grade', action='store_true', help='Whether to calculate accuracy of the predictions.')

    args = parser.parse_args()

    datasets = args.dataset.split(',')
    output = args.output

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                     logprobs=args.num_logprob)

    llm = LLM(model=args.model, trust_remote_code=True)

    datas = from_name(datasets, subset=args.subset)
    for data in datas:
        dataset_name = data['name']
        data = data['data']

        data = preprocess_dataset(data, split=args.split, nshot=args.num_shots > 1, n=args.num_shots)
        prompts = [d['question'] for d in data]

        save_dir = Path(output).expanduser() / dataset_name.replace("/", "_")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f'{args.model.replace("/", "_")}.pkl'

        if args.load and save_path.exists():
            with open(save_path, 'rb') as f:
                outputs = pickle.load(f)
        else:
            outputs = generate_predictions(llm, sampling_params, prompts, save_path, override=args.override)

        if args.perplexity:
            save_path = save_dir / f'{args.model.replace("/", "_")}_ppl.pkl'
            compute_perplexity(outputs, dataset_name, save_path, override=args.override)

        if args.grade:
            save_path = save_dir / f'{args.model.replace("/", "_")}_grades.pkl'
            grade_predictions(outputs, data, dataset_name, save_path, override=args.override)


if __name__ == '__main__':
    main()
