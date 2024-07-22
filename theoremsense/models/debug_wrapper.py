from .model import Model, ChatType, MessageType


class DebugModel(Model):
    def __init__(self):
        super().__init__()
        self.n = 0

    def _generate(self, inputs: list[MessageType], **kwargs):
        self.n += 1
        return f"Debug message {self.n}"

    def _generate_chat(self, inputs: list[MessageType], **kwargs):
        self.n += 1
        return f"Debug chat message {self.n}"