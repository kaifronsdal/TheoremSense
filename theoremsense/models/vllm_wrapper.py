from .model import Model, ChatType, MessageType
from vllm import LLM, SamplingParams
import inspect
import logging
import os
import sys

from multiprocessing import Process, Queue
from tqdm.auto import tqdm


class vLLMBase(Model):

    def __init__(self, model_name: str, tensor_parallel_size: int = 1,
                 trust_remote_code: bool = True, **kwargs):
        """
        A wrapper class for LLM and SamplingParams.

        :param model_name: Huggingface model name or path or checkpoint
        :param tensor_parallel_size: num gpus
        :param trust_remote_code: For downloading huggingface models
        :param kwargs: Extra params for LLM and SamplingParams

        LLM kwargs taken from vLLM docstring:
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq" and "squeezellm". If None, we first check
            the `quantization_config` attribute in the model config file. If
            that is None, we assume the model weights are not quantized and use
            `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig

        SamplingParams kwargs taken from vLLM docstring:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        seed: Random seed to use for the generation.
        use_beam_search: Whether to use beam search instead of sampling.
        length_penalty: Float that penalizes sequences based on their length.
            Used in beam search.
        early_stopping: Controls the stopping condition for beam search. It
            accepts the following values: `True`, where the generation stops as
            soon as there are `best_of` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very
            unlikely to find better candidates; `"never"`, where the beam search
            procedure only stops when there cannot be better candidates
            (canonical beam search algorithm).
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids: List of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        include_stop_str_in_output: Whether to include the stop strings in output
            text. Defaults to False.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
            Note that the implementation follows the OpenAI API: The return
            result includes the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens. The API will always return the
            log probability of the sampled token, so there  may be up to
            `logprobs+1` elements in the response.
        prompt_logprobs: Number of log probabilities to return per prompt token.
        skip_special_tokens: Whether to skip special tokens in the output.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens in the output.  Defaults to True.
        logits_processors: List of functions that modify logits based on
            previously generated tokens.
        """
        super().__init__()
        self.model_name = model_name

        self.model_params = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(LLM.__init__).parameters
        }
        self._sampling_params = {
            k: v for k, v in kwargs.items()
            if k in inspect.signature(SamplingParams.__init__).parameters
        }

        # assert there are no extra params
        assert len(self.model_params) + len(self._sampling_params) == len(kwargs), \
            (f"Extra params found: "
             f"{set(kwargs.keys()) - set(self.model_params.keys()) - set(self._sampling_params.keys())}. "
             f"Make sure you spelled the params correctly and check the docstring for the available params.")

        self.llm = LLM(model_name, trust_remote_code=trust_remote_code, tensor_parallel_size=tensor_parallel_size,
                       **self.model_params)

        self.sampling_params = SamplingParams(**self._sampling_params)

        self.tokenizer = self.llm.get_tokenizer()

    def set_sampling_params(self, **kwargs):
        """
        Set SamplingParams for the model.
        """
        params = {
            k: v for k, v in kwargs.items() if
            k in inspect.signature(SamplingParams.__init__).parameters
        }
        assert len(params) == len(kwargs), \
            (f"Extra params found: "
             f"{set(kwargs.keys()) - set(params.keys())}. "
             f"Make sure you spelled the params correctly and check the docstring for the available params.")
        # override existing params
        self._sampling_params = {**self._sampling_params, **params}
        self.sampling_params = SamplingParams(**self._sampling_params)

    @staticmethod
    def process_outputs(outputs):
        # can be modified to return other things such as logprobs
        return [o.outputs[0].text for o in outputs]

    def _generate(self, inputs: list[MessageType], use_tqdm=True, **kwargs):
        """
        Generate output from inputs.
        """

        outputs = self.llm.generate(inputs, self.sampling_params)
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
        outputs = self.llm.generate(prompts, self.sampling_params)
        outputs = self.process_outputs(outputs)
        return outputs

    def terminate(self):
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        try:
            del self.llm.llm_engine.model_executor.driver_worker
            del self.llm
        except AttributeError:
            pass

        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)


class vLLMProcess(Process):
    def __init__(self, model_name, trust_remote_code=False, tensor_parallel_size=1, device=None, **model_params):
        super().__init__(daemon=True)
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.tensor_parallel_size = tensor_parallel_size
        self.model_params = model_params

        self.device = device

        self.input_queue = Queue()
        self.output_queue = Queue()
        self.model = None
        # Configure logging for capturing subprocess prints
        self.log_queue = Queue()
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.handlers.QueueHandler(self.log_queue))

    def run(self):
        if self.device:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
            self.logger.info(f"Setting CUDA_VISIBLE_DEVICES to {self.device}")

        # Redirect stdout and stderr to logger
        sys.stdout = LoggerWriter(self.logger, logging.INFO)
        sys.stderr = LoggerWriter(self.logger, logging.ERROR)

        self.logger.info("Loading LLM model...")
        self.model = vLLMBase(self.model_name, trust_remote_code=self.trust_remote_code,
                              tensor_parallel_size=self.tensor_parallel_size, **self.model_params)
        self.logger.info("LLM model loaded!")
        while True:
            # Receive data from main process
            next = self.input_queue.get()
            if next is None:
                break
            method_name, args, kwargs = next

            # Call corresponding method on the model
            method = getattr(self.model, method_name)
            result = method(*args, **kwargs)
            # Send results back to main process
            self.output_queue.put(result)

    def terminate(self):
        self.input_queue.put(None)
        self.output_queue.put(None)
        super().terminate()


class vLLM(Model):
    def __init__(self, model_name: str, use_subprocess: bool = False, device=None, trust_remote_code: bool = True,
                 tensor_parallel_size: int = 1, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.tensor_parallel_size = tensor_parallel_size
        self.model_params = kwargs
        self.use_subprocess = use_subprocess
        if use_subprocess:
            print("Starting subprocess")
            self.process = vLLMProcess(model_name, trust_remote_code=trust_remote_code,
                                       tensor_parallel_size=tensor_parallel_size, **kwargs)
            self.process.start()
            self.wait_for_initialization()
        else:
            self.model = vLLMBase(model_name, trust_remote_code=trust_remote_code,
                                  tensor_parallel_size=tensor_parallel_size, **kwargs)

    def wait_for_initialization(self):
        while True:
            log_message = self.process.log_queue.get()
            if "LLM model loaded!" == log_message.getMessage():
                print(f"Subprocess: {log_message.getMessage()}")  # Print the initialization message
                return
            else:
                print(f"Subprocess: {log_message.getMessage()}")  # Print other messages during loading

    def _call_method(self, method_name, *args, **kwargs):
        if self.use_subprocess:
            self.process.input_queue.put((method_name, args, kwargs))
            if method_name in ['generate', 'generate_chat', '__call__']:
                prog_bar = tqdm(total=len(args[0]))
                curr_prog = 0
                while True:
                    log_message = self.process.log_queue.get()
                    if "Processed prompts:" in log_message.getMessage():
                        # look for
                        # "Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s]"
                        # "Processed prompts: 100%|##########| 10/10 [00:28<00:00,  2.88s/it]"
                        # parse out the k/n values
                        k = int(log_message.getMessage().split('/')[0].split()[-1])
                        prog_bar.update(k - curr_prog)
                        curr_prog = k
                    elif log_message.getMessage().strip() != '':
                        tqdm.write(log_message.getMessage())

                    if curr_prog == len(args[0]):
                        break

            return self.process.output_queue.get()
        else:
            method = getattr(self.model, method_name)
            return method(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._call_method('__call__', *args, **kwargs)

    def generate(self, inputs: list[MessageType] | MessageType, **kwargs):
        return self._call_method('generate', inputs, **kwargs)

    def generate_chat(self, inputs: list[ChatType] | ChatType, **kwargs):
        return self._call_method('generate_chat', inputs, **kwargs)

    def set_sampling_params(self, **kwargs):
        return self._call_method('set_sampling_params', **kwargs)

    def terminate(self):
        if self.use_subprocess:
            self.process.terminate()
        else:
            self.model.terminate()
