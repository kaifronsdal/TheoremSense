"""
Takes in model and dataset, and generates predictions for the dataset. Saves the predictions to a file.
"""

# python theoremsense/generate_predictions.py --model=deepseek-ai/deepseek-math-7b-instruct --dataset=EleutherAI/hendrycks_math --subset=algbra --output=~/generated_outputs --perplexity=True --num_shots=3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # "2,3,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import argparse
from dataset import from_name
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import pipeline

from prompt import generate_nshot_prompts
from grader import ExactMatchGrader, NextTokenAccuracyGrader
from dataset import get_boxed_answer
from latex_formater import latex_deformat
import torch

from pathlib import Path
from tqdm import tqdm
from metrics import teacher_forcing_predictions
import numpy as np
import pickle

DEVICE = "cuda"


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


def generate_predictions(prompts, pipe, output_path, override=False, generate_config=None):
    if Path(output_path).exists() and not override:
        raise ValueError(f'Output file {output_path} already exists. Use --override to overwrite it.')

    # outputs = []
    # for i in tqdm(range(0, len(prompts), batch_size)):
    #     batch_prompts = prompts[i:i + batch_size]
    #     # tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    #     # output = model.generate(tokens.input_ids.to(DEVICE), attention_mask=tokens.attention_mask.to(DEVICE),
    #     #                         generation_config=generate_config)
    #     # output = tokenizer.batch_decode(output, skip_special_tokens=True)
    #     # remove prompt from output
    #     output = [o[len(p):] for o, p in zip(output, batch_prompts)]
    #     outputs.extend(output)

    chunk_size = generate_config.get('batch_size', 8) * 4

    outputs = []
    for i in tqdm(range(0, len(prompts), chunk_size)):
        batch_prompts = prompts[i:i + chunk_size]
        batch_outputs = pipe(batch_prompts, **generate_config)
        outputs.extend(batch_outputs)

    # outputs = [o for o in tqdm(pipe(prompts, **generate_config))]

    with open(output_path, 'wb') as f:
        pickle.dump(outputs, f)

    return outputs


def grade_predictions(outputs, data, dataset, save_path, override=False):
    if Path(save_path).exists() and not override:
        raise ValueError(f'Output file {save_path} already exists. Use --override to overwrite it.')

    grader = ExactMatchGrader()

    boxed_predictions = [latex_deformat(get_boxed_answer(o)) for o in outputs]
    boxed_answers = [latex_deformat(d['boxed']) for d in data]
    grades = grader.grade(boxed_predictions, boxed_answers)

    print(f'Total accuracy for {dataset} is {np.mean(grades)}.')

    with open(save_path, 'wb') as f:
        pickle.dump(grades, f)
    return grades


# create a batched version of generate_prediction
def batched_teacher_forcing_predictions(prompts, solutions, model, tokenizer, debug=False):
    input = tokenizer([f"{p}\n\n{s}" for p, s in zip(prompts, solutions)], return_tensors="pt", padding=True,
                      truncation=True)
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        response = model(input.input_ids.to(DEVICE), attention_mask=input.attention_mask.to(DEVICE))
    # get the response ids for the solution part of the input
    all_preds = torch.argmax(response.logits, dim=-1).cpu()

    tf_ids = []
    solution_ids = []
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

    if debug:
        for i in range(len(tf_ids)):
            print(f"Prompt: {prompts[i]}")
            assert len(tf_ids[i]) == len(solution_ids[i])
            for j in range(len(tf_ids[i])):
                print(f"Predicted:\t{tokenizer.decode([tf_ids[i][j]], skip_special_tokens=True).__repr__()}")
                print(f"Actual:   \t{tokenizer.decode(solution_ids[i][j], skip_special_tokens=True).__repr__()}")

    return tf_ids, solution_ids


def generate_teacher_forcing_predictions(prompts, solutions, model, tokenizer, save_path, override=False, batch_size=8,
                                         debug=False):
    if Path(save_path).exists() and not override:
        raise ValueError(f'Output file {save_path} already exists. Use --override to overwrite it.')

    teacher_forced_predictions = []
    solution_ids = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_solutions = solutions[i:i + batch_size]
        tf_ids, sol_ids = batched_teacher_forcing_predictions(batch_prompts, batch_solutions, model, tokenizer, debug)
        teacher_forced_predictions.extend(tf_ids)
        solution_ids.extend(sol_ids)

    with open(save_path, 'wb') as f:
        pickle.dump((teacher_forced_predictions, solution_ids), f)

    return teacher_forced_predictions, solution_ids


def compute_tfa(tf_pred_ids, solution_ids, save_path, override=False):
    if Path(save_path).exists() and not override:
        raise ValueError(f'Output file {save_path} already exists. Use --override to overwrite it.')

    tfa = []
    for i in range(len(tf_pred_ids)):
        # compute accuracy for each prompt
        acc = np.mean([
            tf_pred_ids[i][j] == solution_ids[i][j]
            for j in range(len(tf_pred_ids[i]))
        ])
        tfa.append(acc)

    with open(save_path, 'wb') as f:
        pickle.dump(tfa, f)

    return tfa


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
    parser.add_argument('--override', action='store_true',
                        help='Whether to override the output file if it exists.')
    parser.add_argument('--load', action='store_true',
                        help='Whether to load predictions from file if it exists.')

    # Sampling parameters
    parser.add_argument('--no_sample', action='store_false', help='Whether to use sampling for generation.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling.')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate.')
    parser.add_argument('--num_shots', type=int, default=0, help='Number of shots to generate')

    # post processing parameters
    parser.add_argument('--tfa', action='store_true',
                        help='Whether to calculate teacher forcing accuracy of the predictions.')
    parser.add_argument('--grade', action='store_true',
                        help='Whether to calculate normal boxed accuracy on autoregressive generations.')

    args = parser.parse_args()

    datasets = args.dataset.split(',')
    output = args.output

    # sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
    #                                  logprobs=args.num_logprob)

    # llm = LLM(model=args.model, trust_remote_code=True)

    pipe = pipeline('text-generation', model=args.model, tokenizer=args.model, device_map="auto",
                    torch_dtype=torch.bfloat16, batch_size=args.batch_size)
    tokenizer = pipe.tokenizer
    model = pipe.model
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenizer.pad_token_id = pad_token_id

    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    # pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.no_sample:
        generate_config = dict(do_sample=False, max_new_tokens=args.max_tokens, return_full_text=False,
                               pad_token_id=pad_token_id, num_return_sequences=1)
    else:
        generate_config = dict(do_sample=True, max_new_tokens=args.max_tokens, return_full_text=False,
                               top_p=args.top_p, temperature=args.temperature,
                               pad_token_id=pad_token_id, num_return_sequences=1)

    datas = from_name(datasets, subset=args.subset)
    for data in datas:
        dataset_name = data['name']
        data = data['data']

        data = preprocess_dataset(data, split=args.split, nshot=args.num_shots > 1, n=args.num_shots)
        prompts = [d['question'] for d in data]
        solutions = [d['answer'] for d in data]

        save_dir = Path(output).expanduser() / dataset_name.replace("/", "_")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f'{args.model.replace("/", "_")}_teacher_forcing.pkl'

        if args.load and save_path.exists():
            with open(save_path, 'rb') as f:
                tf_outputs = pickle.load(f)
        else:
            tf_outputs = generate_teacher_forcing_predictions(prompts, solutions, model, tokenizer,
                                                              batch_size=args.batch_size, save_path=save_path,
                                                              override=args.override)

        if args.tfa:
            save_path = save_dir / f'{args.model.replace("/", "_")}_teacher_forcing_accuracy.pkl'
            tfa = compute_tfa(tf_outputs[0], tf_outputs[1], save_path, override=args.override)
            print(f"Mean TFA: {np.mean(tfa)}")

        if args.grade:
            save_path = save_dir / f'{args.model.replace("/", "_")}_predictions.pkl'
            if args.load and save_path.exists():
                with open(save_path, 'rb') as f:
                    outputs = pickle.load(f)
            else:
                outputs = generate_predictions(prompts, pipe, save_path, override=args.override,
                                               generate_config=generate_config)
            save_path = save_dir / f'{args.model.replace("/", "_")}_grades.pkl'
            grades = grade_predictions(outputs, data, dataset_name, save_path, override=args.override)
            print(f"Mean grade: {np.mean(grades)}")


if __name__ == '__main__':
    main()
