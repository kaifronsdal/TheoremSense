import math
from copy import deepcopy
from typing import Iterable

ChatType = list[dict[str, str]] | list[str]
MessageType = str


def is_chat_template_format(data: ChatType) -> bool:
    """
    Check if the data is in chat format.
    """
    if not isinstance(data, list):
        return False

    if len(data) == 0:
        return True

    return isinstance(data[0], dict)


def validate_history(history: list[dict[str, str]] | list[str]):
    """
    Validate chat history.
    """
    assert isinstance(history, list), 'Chat history should be a list.'
    # assert len(history) % 2 == 0, 'Chat history should alternate between user and assistant messages.'
    if len(history) == 0:
        return

    message_type = type(history[0])

    for i, message in enumerate(history):
        if message_type is str:
            assert isinstance(message, str), \
                'Each message in the chat should match the first message which is a string.'
        else:
            assert isinstance(message, dict), \
                'Each message in the chat should match the first message which is a dictionary.'
            assert 'role' in message, 'Role key not found in message.'
            assert 'content' in message, 'Content key not found in message.'
            if i % 2 == 0:
                assert message['role'] == 'user', 'User message should be at even index.'
            else:
                assert message['role'] == 'assistant', 'Assistant message should be at odd index.'


def standardize_input(input: list[MessageType] | MessageType | list[ChatType] | ChatType):
    """
    Standardize input to a list of messages.
    """
    is_singleton = False
    if isinstance(input, MessageType) or (isinstance(input, list) and isinstance(input[0], dict)):
        input = [input]
        is_singleton = True

    for i in range(len(input)):
        assert isinstance(input[i], str) or isinstance(input[i], list), \
            'Input must be a list of messages, strings, or Prompt objects that generate a list of messages or strings.'

        if isinstance(input[i], list):
            validate_history(input[i])

    return input, is_singleton
