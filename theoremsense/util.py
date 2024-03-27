from pathlib import Path
import pickle
from typing import Callable


# decorator to save the output of the function
def save_output(func):
    def wrapper(*args, output_path=None, override=False, load=False, **kwargs):
        if output_path is not None:
            if Path(output_path).exists():
                if load:
                    print(f'Loading output from {output_path}')
                    with open(output_path, 'rb') as f:
                        return pickle.load(f)
                if not override:
                    raise ValueError(f'Output file {output_path} already exists. Use --override to overwrite it.')

            output = func(*args, **kwargs)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(output, f)
            return output
        else:
            return func(*args, **kwargs)

    return wrapper


class Prompt(list):
    """
    A class to represent a prompt.The same as a list, but used to control flattening of nested lists.
    """

    def __init__(self, *args):
        super().__init__(*args)


def flatten_array(nested_structure):
    flatten = []

    if isinstance(nested_structure, dict):
        for key, value in nested_structure.items():
            flatten.extend(flatten_array(value))
    elif isinstance(nested_structure, list) and not isinstance(nested_structure, Prompt):
        for sub in nested_structure:
            flatten.extend(flatten_array(sub))
    else:
        flatten.append(nested_structure)

    return flatten


def _unflatten_array(flat_array, prototype):
    unflatten = None
    i = 0

    if isinstance(prototype, dict):
        unflatten = {}
        for key, value_prototype in prototype.items():
            value_unflatten, j = _unflatten_array(flat_array[i:], value_prototype)
            i += j
            unflatten[key] = value_unflatten
    elif isinstance(prototype, list) and not isinstance(prototype, Prompt):
        unflatten = []
        for sub in prototype:
            sub_unflatten, j = _unflatten_array(flat_array[i:], sub)
            i += j
            unflatten.append(sub_unflatten)
    else:
        unflatten = flat_array[i]
        i += 1

    return unflatten, i


def unflatten_array(flat_array, prototype):
    unflatten, _ = _unflatten_array(flat_array, prototype)
    return unflatten


def generate(inputs: list[str], model: Callable, filter: Callable, max_attempts: int | None = 10) -> list[str]:
    """
    Generates outputs using the model callable. The outputs are then filtered using the filter function. The outputs that do not pass the filter are then re-generated until they pass the filter. The outputs are then returned.
    :param inputs: The inputs to the model. Can be a list of strings or a nested list of strings. Output will be the same shape as the input.
    :param model: The model callable. This should take a list of inputs and return a list of outputs.
    :param filter: The filter function. This should take a single output and return a boolean indicating whether the output is good or not, and a filtered version of the output.
    :param max_attempts: The maximum number of attempts to generate a valid output. If None, there is no limit.
    :return: The filtered outputs. The shape of the output will be the same as the input.

    Example vLLM:
    >>> from vllm import LLM, SamplingParams
    >>> sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
    >>> llm = LLM(model=model_name, trust_remote_code=True)
    >>> def vllm_generate(inputs):
    ...     return [x.outputs[0].text for x in llm.generate(inputs, sampling_params)]
    >>> def filter(response):
    ...     prediction  = response[0]
    ...     return prediction.isdigit(), prediction
    >>> questions = ['What is 2 + 2?', 'What is 3 + 3?']
    >>> generate(questions, vllm_generate, filter)
    ['4', '6']
    Example Google-API:
    >>> import google.generativeai as genai
    >>> import os
    >>> genai.configure(api_key=os.environ["GOOGLE_AI_API_KEY"])
    >>> model = genai.GenerativeModel('gemini-pro')
    >>> def google_generate(inputs):
    ...     return [model.generate_content(input).text for input in inputs]
    >>> def filter(response):
    ...     prediction  = response[0]
    ...     return prediction.isdigit(), prediction
    >>> questions = ['What is 2 + 2?', 'What is 3 + 3?']
    >>> generate(questions, google_generate, filter)
    ['4', '6']
    """
    inputs_flat = flatten_array(inputs)
    remaining_idxs = list(range(len(inputs_flat)))
    outputs = [None] * len(inputs_flat)
    attempts = 0

    while len(remaining_idxs) > 0 and (max_attempts is None or attempts < max_attempts):
        attempts += 1
        new_remaining_idxs = []
        new_outputs = model([inputs_flat[i] for i in remaining_idxs])

        for i, output in zip(remaining_idxs, new_outputs):
            good, filtered = filter(output)
            if good:
                outputs[i] = filtered
            else:
                new_remaining_idxs.append(i)

        remaining_idxs = new_remaining_idxs
    return unflatten_array(outputs, inputs)


def filter_errors(func, func_is_filter: bool = False):
    """
    Decorator to catch exceptions and return a boolean indicating success and the result of the function. Used to
    filter out errors from the function.
    :param func: The function to wrap.
    :param func_is_filter: Whether the function is a filter function. If True, the function should return a tuple with a
    boolean indicating success and the result of the function. If False, the function should return the result of the
    function.
    :return: A wrapped function that returns a tuple with a boolean indicating success and the result of the function.
    """

    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if func_is_filter:
                return result
            else:
                return True, result
        except Exception as e:
            return False, None

    return wrapper


if __name__ == '__main__':
    nested_array = [[1, 2, 3, [4, 5, 6]], [7, 8, 9], [[10, 11], 12, 13]]
    assert unflatten_array(flatten_array(nested_array), nested_array) == nested_array

    nested_array = []
    assert unflatten_array(flatten_array(nested_array), nested_array) == nested_array

    nested_array = [1, 2, 3]
    assert unflatten_array(flatten_array(nested_array), nested_array) == nested_array

    nested_array = [[1, 2, 3], [4, 5, 6], [7, [8, 9]]]
    assert unflatten_array(flatten_array(nested_array), nested_array) == nested_array

    nested_array = [1, [2, [3, 4], 5], 6]
    assert unflatten_array(flatten_array(nested_array), nested_array) == nested_array

    nested_structure = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
    assert unflatten_array(flatten_array(nested_structure), nested_structure) == nested_structure

    nested_structure = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4, 'f': {'g': 5, 'h': 6}}}
    assert unflatten_array(flatten_array(nested_structure), nested_structure) == nested_structure

    nested_mixed = {'a': 1, 'b': [2, 3], 'c': {'d': 4, 'e': [5, 6]}}
    assert unflatten_array(flatten_array(nested_mixed), nested_mixed) == nested_mixed

    nested_mixed = {'a': 1, 'b': [2, 3], 'c': {'d': 4, 'e': [5, 6, {'f': 7, 'g': 8}]}}
    assert unflatten_array(flatten_array(nested_mixed), nested_mixed) == nested_mixed
