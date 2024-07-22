from .util import standardize_input, MessageType, ChatType


def get_input_type(inputs):
    if isinstance(inputs[0], str):
        assert all(isinstance(x, str) for x in inputs), \
            'Invalid input type. All elements should be either MessageType or ChatType'
        return "message_type"
    elif isinstance(inputs[0], list) and isinstance(inputs[0][0], dict):
        assert all(isinstance(x, list) and isinstance(x[0], dict) for x in inputs), \
            'Invalid input type. All elements should be either MessageType or ChatType'
        return "chat_type"
    else:
        raise ValueError(f'Invalid input type: {type(inputs[0])}\n{inputs[0]}')


class Model:
    """
    Model interface for inference. Subclasses should implement the _generate and _generate_chat methods.
    """

    def __init__(self):
        pass

    def _generate(self, inputs: list[MessageType], **kwargs):
        """
        Generate output from inputs. This method should be implemented by the subclass.
        """
        raise NotImplementedError

    def generate(self, inputs: list[MessageType] | MessageType, **kwargs):
        """
        Generate output from inputs.
        """
        inputs, is_singleton = standardize_input(inputs)
        response = self._generate(inputs, **kwargs)
        return response[0] if is_singleton else response

    def _generate_chat(self, inputs: list[ChatType], **kwargs):
        """
        Generate chat response from input. Chats are usually generated in a conversational context and are of the form
        [
            {'role': 'user', 'text': 'Hello!'},
            {'role': 'assistant', 'text': 'Hi! How can I help you?'}
            ...
        ]
        This method should be implemented by the subclass.
        """
        raise NotImplementedError

    def generate_chat(self, inputs: list[ChatType] | ChatType, **kwargs):
        """
        Generate chat response from input. Chats are usually generated in a conversational context and are of the form
        [
            {'role': 'user', 'text': 'Hello!'},
            {'role': 'assistant', 'text': 'Hi! How can I help you?'}
            ...
        ]
        """
        inputs, is_singleton = standardize_input(inputs)
        response = self._generate_chat(inputs, **kwargs)
        return response[0] if is_singleton else response

    def __call__(self, inputs: list[MessageType] | MessageType | list[ChatType] | ChatType, **kwargs):
        """
        Generate output from inputs. If inputs is a list, generate output for each element in the list.
        """
        inputs, is_singleton = standardize_input(inputs)
        if len(inputs) == 0:
            raise ValueError('Input list is empty')

        input_type = get_input_type(inputs)
        if input_type == "message_type":
            result = self._generate(inputs, **kwargs)
        elif input_type == "chat_type":
            result = self._generate_chat(inputs, **kwargs)
        else:
            raise ValueError(f'Invalid input type: {type(inputs[0])}\n{inputs[0]}')

        return result[0] if is_singleton else result