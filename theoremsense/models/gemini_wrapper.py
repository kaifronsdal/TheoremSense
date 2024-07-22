from .model import Model, ChatType, MessageType

import google.generativeai as genai
import time
from tqdm.auto import tqdm
import os

DEFAULT_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]


def openai_to_google(messages):
    """
    Converts OpenAI chat messages to the Google chat message format.
    :param messages: OpenAI chat messages.
    :return: Google chat messages.
    """
    return [
        {
            "role": "user" if message['role'] == "user" else "model",
            "parts": [message['content']]
        }
        for message in messages
    ]


import traceback


def _generate(model, input: MessageType | ChatType, sleep_time=10 * 60):
    """
    Generate output from a single input
    :param model: The model to use.
    :param input: The input to generate output from.
    :param sleep_time: The time to sleep if the rate limit is hit. Default is 10 minutes.
    """
    if isinstance(input, list):
        input = openai_to_google(input)

    # TODO: better error types
    try:
        result = model.generate_content(input)
    except Exception as e:
        print(f'Error generating content: {traceback.format_exc()}')
        # sleep for if rate limit is hit
        # TODO: what's the best sleep amount?
        print(f'Rate limit hit. Sleeping for {sleep_time / 60} minutes.')
        time.sleep(sleep_time)
        result = None

    # if the result was filtered by google's safety settings, return empty string
    try:
        return result.text
    except Exception as e:
        return ''


class Gemini(Model):
    def __init__(self, model_name='gemini-1.0-pro', api_key=None, safety_settings=None, sleep_time=10 * 60,
                 **generation_config):
        """
        Initialize the Gemini model.
        :param model_name: The name of the model to use. i.e.
        :param api_key: Google API key to use, overrides any key set in the environment.
        :param safety_settings: Safety settings to use. If None, use the default safety settings.
        :param sleep_time: The time to sleep in seconds if the rate limit is hit. Default is 10 minutes.
        :param kwargs: Generation parameters to pass to the model.
            from https://ai.google.dev/api/python/google/generativeai/GenerationConfig
            candidate_count: (int | None)
            stop_sequences: (Iterable[str] | None)
            max_output_tokens: (int | None)
            temperature: (float | None)
            top_p: (float | None)
            top_k: (int | None)
        """
        super().__init__()
        env_google_api_key = os.getenv("GOOGLE_AI_API_KEY")
        assert env_google_api_key or api_key, \
            ("API key not set. Please set the GOOGLE_AI_API_KEY environment variable or pass "
             "an api_key to the Gemini constructor.")
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=env_google_api_key)

        self.model_name = model_name
        self.generation_config = generation_config
        self.safety_settings = safety_settings if safety_settings else DEFAULT_SAFETY_SETTINGS
        self.model = genai.GenerativeModel(model_name=model_name,
                                           generation_config=generation_config,
                                           safety_settings=safety_settings)

        self.sleep_time = sleep_time

    def _generate(self, inputs: list[MessageType], use_tqdm=True, **kwargs):
        """
        Generate output from inputs.
        """

        if use_tqdm:
            return [
                _generate(self.model, input, sleep_time=self.sleep_time)
                for input in tqdm(inputs, desc='Generating', unit='message')
            ]
        else:
            return [
                _generate(self.model, input, sleep_time=self.sleep_time)
                for input in inputs
            ]

    def _generate_chat(self, inputs: list[ChatType], use_tqdm=True, **kwargs):
        """
        Generate chat response from input. Chats are usually generated in a conversational context and are of the form
        [
            {'role': 'user', 'text': 'Hello!'},
            {'role': 'assistant', 'text': 'Hi! How can I help you?'}
            ...
        ]
        """

        return self._generate(inputs, **kwargs)