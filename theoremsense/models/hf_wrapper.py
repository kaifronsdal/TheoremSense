from .model import Model, ChatType, MessageType, get_input_type
from .util import validate_history, standardize_input
from transformers import pipeline
import inspect
import torch
from typing import Literal
from tqdm.auto import tqdm

MethodType = Literal['autoregressive', 'teacher_forcing']

TeacherForcingType = dict[str]


def batched_teacher_forcing_predictions(prompts, solutions, model, tokenizer, device, debug=False):
    prompt_input_type = get_input_type(prompts[0])
    solution_input_type = get_input_type(solutions[0])
    if prompt_input_type == "message_type":
        assert solution_input_type == "message_type"
        prompts_and_solutions = [f"{p}\n\n{s.strip()}" for p, s in zip(prompts, solutions)]
        prompts = [f"{p}\n\n" for p in prompts]
    elif prompt_input_type == "chat_type":
        if solution_input_type == "message_type":
            solutions = [{'role': 'assistant', 'content': s} for s in solutions]

        prompts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in
                   prompts]
        prompts_and_solutions = [
            tokenizer.apply_chat_template(p + s, tokenize=False, add_generation_prompt=True)
            for p, s in zip(prompts, solutions)
        ]
    else:
        raise ValueError(f"Unsupported input type: {prompt_input_type}")

    input = tokenizer(prompts_and_solutions, return_tensors="pt", padding=True, truncation=True)
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    # compute num chars
    total_num_chars = [len(ps) for ps in prompts_and_solutions]
    prompt_num_chars = [len(p) for p in prompts]
    solution_num_chars = [t - p for t, p in zip(total_num_chars, prompt_num_chars)]

    with torch.no_grad():
        response = model(input.input_ids.to(device), attention_mask=input.attention_mask.to(device))
    # get the response ids for the solution part of the input
    all_preds = torch.argmax(response.logits, dim=-1).cpu()
    all_logits = response.logits.cpu()

    results = []
    for i in range(len(prompts)):
        # assumes attention mask is contiguous which I think is reasonable for usual inputs
        total_length = input.attention_mask[i].sum().item()
        prompt_start = torch.nonzero(prompt_tokens.attention_mask[i, :] == 1, as_tuple=False)[0, 0].item()
        prompt_length = prompt_tokens.attention_mask[i].sum().item()
        prompt_end = prompt_start + prompt_length
        input_start = torch.nonzero(input.attention_mask[i, :] == 1, as_tuple=False)[0, 0].item()
        solution_start = input_start + prompt_length
        solution_length = total_length - prompt_length

        # ensure that the last 3 tokens of the prompt are the same as the last 3 tokens before the solution
        try:
            # the last token can sometimes combine with characters from the solution
            # assert prompt_tokens.input_ids[i, -1] == input.input_ids[i, solution_start - 1]
            assert prompt_tokens.input_ids[i, prompt_end - 2] == input.input_ids[i, solution_start - 2]
            assert prompt_tokens.input_ids[i, prompt_end - 3] == input.input_ids[i, solution_start - 3]
            # make sure mask_end is correct
            assert input_start == 0 or input.attention_mask[i, input_start - 1] == 0
            assert input.attention_mask[i, input_start] == 1
        except AssertionError:
            print(prompt_tokens.input_ids[i, -1])
            print(tokenizer.decode([prompt_tokens.input_ids[i, -1]]))
            print(input.input_ids[i, solution_start - 1])
            print(tokenizer.decode([input.input_ids[i, solution_start - 1]]))
            # print(f"Prompt: {prompts[i]}")
            # print(f"Prompt tokens: {prompt_tokens.input_ids[i]}")
            # print(f"Attention mask: {input.attention_mask[i]}")
            # print(f"Input: {input.input_ids[i]}")
            # print(f"Solution start: {solution_start}")
            # print(f"Mask end: {mask_end}")
            # print(f"Total length: {total_length}")
            # print(f"Prompt length: {prompt_length}")
            # print(f"Solution length: {solution_length}")
            raise

        # print(input.attention_mask[i])
        # print(f'prompt_tokens[i]: {[tokenizer.decode([t]) for t in prompt_tokens.input_ids[i]]}')
        # print(f'input[i]: {[tokenizer.decode([t]) for t in input.input_ids[i]]}')
        # print(f'all_preds[i]: {[tokenizer.decode([t]) for t in all_preds[i]]}')
        # print(
        #     f'solution[i]: {[tokenizer.decode([t]) for t in input.input_ids[i, solution_start:solution_start + solution_length]]}')
        # print(
        #     f'predicted: {[tokenizer.decode([t]) for t in all_preds[i, solution_start - 1:solution_start - 1 + solution_length]]}')
        #
        # raise ValueError

        tf_id = all_preds[i, solution_start - 1:solution_start - 1 + solution_length]
        solution_id = input.input_ids[i, solution_start:solution_start + solution_length]
        tf_logit = all_logits[i, solution_start - 1:solution_start - 1 + solution_length]

        loss = torch.nn.functional.cross_entropy(tf_logit, solution_id).item()
        results.append({
            'teacher_forced_ids': tf_id,
            'solution_ids': solution_id,
            'teacher_forced_logits': tf_logit,
            'loss': loss,
            'total_num_tokens': total_length,
            'prompt_num_tokens': prompt_length,
            'solution_num_tokens': solution_length,
            'total_num_chars': total_num_chars[i],
            'prompt_num_chars': prompt_num_chars[i],
            'solution_num_chars': solution_num_chars[i]
        })

    if debug:
        for i in range(len(prompts)):
            print(f"Prompt: {prompts[i]}")
            assert len(results[i]['teacher_forced_ids']) == len(results[i]['solution_ids'])
            for j in range(len(results[i]['teacher_forced_ids'])):
                print(
                    f"Predicted:\t{tokenizer.decode(results[i]['teacher_forced_ids'][j], skip_special_tokens=True).__repr__()}")
                print(
                    f"Actual:   \t{tokenizer.decode(results[i]['solution_ids'][j], skip_special_tokens=True).__repr__()}")

    return results


def generate_teacher_forcing(prompts, solutions, model, tokenizer,
                             batch_size=8, device=None, debug=False):
    if device is None:
        device = model.device
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]
        batch_solutions = solutions[i:i + batch_size]
        batch_results = batched_teacher_forcing_predictions(batch_prompts, batch_solutions, model, tokenizer, device, debug)
        results.extend(batch_results)

    return results


class HFModel(Model):

    def __init__(self, model_name, torch_dtype=torch.bfloat16, batch_size=8,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size

        self.model_params = {
            k: v for k, v in kwargs.items() if k in inspect.signature(pipeline.__init__).parameters
        }

        # assert there are no extra params
        # assert len(self.model_params) == len(kwargs), \
        #     (f"Extra params found: "
        #      f"{set(kwargs.keys()) - set(self.model_params.keys())}. "
        #      f"Make sure you spelled the params correctly and check the docstring for the available params.")

        self.model = pipeline('text-generation', model=model_name, tokenizer=model_name, device_map="auto",
                              torch_dtype=torch_dtype, batch_size=batch_size, **self.model_params)

        tokenizer = self.model.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_token_id
        self.tokenizer = tokenizer

        base_model = self.model.model
        if hasattr(base_model, 'generation_config') and base_model.generation_config.pad_token_id is None:
            base_model.generation_config.pad_token_id = base_model.generation_config.eos_token_id
        self.base_model = base_model

    @staticmethod
    def process_outputs(outputs):
        return [o[0]['generated_text'] for o in outputs]

    def _generate(self, inputs: list[MessageType], use_tqdm=True, **kwargs):
        """
        Generate output from inputs.
        """
        outputs = self.model(inputs)
        outputs = self.process_outputs(outputs)
        return outputs

    def _generate_chat(self, inputs: list[ChatType], use_tqdm=True, **kwargs):
        """
        Generate chat response from input. Chats are usually generated in a conversational context and are of the form
        [
            {'role': 'user', 'content': 'Hello!'},
            {'role': 'assistant', 'content': 'Hi! How can I help you?'}
            ...
        ]
        """

        prompts = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in
                   inputs]
        outputs = self.model(prompts)
        outputs = self.process_outputs(outputs)
        return outputs

    def generate_teacher_forcing(self, inputs: TeacherForcingType | list[TeacherForcingType], **kwargs):
        """
        Generate teacher-forced predictions for the given inputs. The inputs can be a list of messages or a dictionary
        with prompts and solutions.
        """
        if isinstance(inputs, dict):
            inputs = [inputs]

        for i in range(len(inputs)):
            question, is_singleton = standardize_input(inputs[i]['question'])
            assert is_singleton, "Teacher forcing only supports single messages as prompts."
            solution, is_singleton = standardize_input(inputs[i]['solution'])
            assert is_singleton, "Teacher forcing only supports single messages as solutions."

            inputs[i] = {'question': question[0], 'solution': solution[0]}

        prompts = [i['question'] for i in inputs]
        solutions = [i['solution'] for i in inputs]

        return generate_teacher_forcing(prompts, solutions, self.base_model, self.tokenizer, self.batch_size, **kwargs)