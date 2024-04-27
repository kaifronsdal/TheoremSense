from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from vllm import SamplingParams


def perplexity(outputs):
    if not isinstance(outputs, list):
        outputs = [outputs]

    ppls = []
    loss_fn = CrossEntropyLoss(reduction='none')
    for o in outputs:
        # o.logprobs is a dict from token ids to log probabilities
        # o.token_ids is a sequence of tokens
        # create a tensor of token ids of shape (seq_len, k)
        # ensure that the target tokens are the first tokens in the logprobs array
        assert [list(v.keys())[0] for v in o.logprobs] == o.token_ids

        logprobs = [list(v.values()) for v in o.logprobs]
        # pad logprobs with -inf to ensure that the tensor is of shape (seq_len, k)
        max_len = max(len(l) for l in logprobs)
        logprobs = [l + [-float('inf')] * (max_len - len(l)) for l in logprobs]
        logprobs = torch.tensor(logprobs)

        ppl = np.exp(np.mean(loss_fn(logprobs, torch.zeros(logprobs.shape[0], dtype=torch.long)).tolist()))
        ppls.append(ppl)

    return ppls


def generate_teacher_forced_prompts(prompt, solution, model):
    """
    Generate teacher-forced prompts for a given prompt and solution.

    :param prompt: The prompt to use.
    :param solution: The solution to use.
    :return: A list of prompts, each with one additional token from the solution, and the end position of the prefix in the solution.
    """
    prefix_pos = len(model.llm_engine.tokenizer.encode(prompt)) - 1
    token_ids = model.llm_engine.tokenizer.encode(solution, add_special_tokens=False)
    solution_tokens = model.llm_engine.tokenizer.convert_ids_to_tokens(token_ids)
    # get cumulative sum of lengths of tokens
    prompt_lengths = np.cumsum([0] + [len(t) for t in solution_tokens])
    assert prompt_lengths[-1] == len(solution)
    prompts = [prompt + solution[:prompt_lengths[i]] for i in range(len(prompt_lengths[:-1]))]
    return prompts, token_ids, prefix_pos


def teacher_forcing_predictions(prompt, solution, model):
    """
    Generate teacher-forced predictions for a given prompt and solution.

    :param prompt: The prompt to use.
    :param solution: The solution to use.
    :param model: The model to use.
    :return: The predicted tokens, the token ids of the solution, and the position of the prefix in the solution.
    """
    prompts, token_ids, prefix_pos = generate_teacher_forced_prompts(prompt, solution, model)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    outputs = model.generate(prompts, sampling_params)
    return outputs, token_ids, prefix_pos


def teacher_forcing_accuracy(prompt, solution, model, debug=False):
    """
    Calculate the average per-token accuracy for a given prompt and solution using teacher forcing.

    The average per-token accuracy is calculated for a ground truth answer (t1,...,tn) by averaging
    the next token accuracies conditioned on the ground truth (i.e. 1 if the highest-probability token
    predicted by the model conditioned on (t1,...,tk) matches the ground truth for all 1 <= k <= {n-1}).

    :param prompt: The prompt to use.
    :param solution: The solution to use.
    :param model: The model to use.
    :param debug: Whether to print debug information (predicted vs ground truth tokens for each token in the solution)
    :return: The average per-token accuracy.
    """
    outputs, token_ids, _ = teacher_forcing_predictions(prompt, solution, model)
    accuracies = []
    predicted_tokens = []
    for i, o in enumerate(outputs):
        # get the token id of the predicted token
        pred_token_id = o.outputs[0].token_ids[0]
        # get the token id of the correct token
        correct_token_id = token_ids[i]
        # check if the predicted token is correct
        accuracies.append(int(pred_token_id == correct_token_id))
        # store the predicted token
        predicted_tokens.append(pred_token_id)
        if debug:
            print(f"Predicted:\t\"{model.llm_engine.tokenizer.convert_ids_to_tokens([pred_token_id])[0]}\"")
            print(f"Correct:\t\"{model.llm_engine.tokenizer.convert_ids_to_tokens([correct_token_id])[0]}\"")

    return np.mean(accuracies), predicted_tokens

# def teacher_forcing_accuracy(prompt, solution, model, debug=False):
#     """
#     Calculate the average per-token accuracy for a given prompt and solution using teacher forcing.
#
#     The average per-token accuracy is calculated for a ground truth answer (t1,...,tn) by averaging
#     the next token accuracies conditioned on the ground truth (i.e. 1 if the highest-probability token
#     predicted by the model conditioned on (t1,...,tk) matches the ground truth for all 1 <= k <= {n-1}).
#
#     :param prompt: The prompt to use.
#     :param solution: The solution to use.
#     :param model: The model to use.
#     :param debug: Whether to print debug information (predicted vs ground truth tokens for each token in the solution)
#     :return: The average per-token accuracy.
#     """
#     prompts, token_ids, prefix_pos = generate_teacher_forced_prompts(prompt, solution, model)
#     sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
#     outputs = model.generate(prompts, sampling_params)
#     accuracies = []
#     predicted_tokens = []
#     for i, o in enumerate(outputs):
#         # get the token id of the predicted token
#         pred_token_id = o.outputs[0].token_ids[0]
#         # get the token id of the correct token
#         correct_token_id = token_ids[i]
#         # check if the predicted token is correct
#         accuracies.append(int(pred_token_id == correct_token_id))
#         # store the predicted token
#         predicted_tokens.append(pred_token_id)
#         if debug:
#             print(f"Predicted:\t\"{model.llm_engine.tokenizer.convert_ids_to_tokens([pred_token_id])[0]}\"")
#             print(f"Correct:\t\"{model.llm_engine.tokenizer.convert_ids_to_tokens([correct_token_id])[0]}\"")
#
#     return np.mean(accuracies), predicted_tokens
