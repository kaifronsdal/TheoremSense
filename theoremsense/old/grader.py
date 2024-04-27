"""
Contains implementations of various proxy metrics for measuring the correctness of a theorem proof.
Currently implemented:
To implement:
    - Exact Match
    - Next-Token Prediction Accuracy
    - AI prediction
    - (boxed answer) Prompt Length
"""

import numpy as np


class BaseGrader:

    def grade(self, answers: list[str], targets: list[str]) -> list:
        raise NotImplementedError


class ExactMatchGrader(BaseGrader):
    """
    Computes the exact match accuracy of the proof.
    """

    def grade(self, answers: list[str], targets: list[str]) -> list:
        return [a == t for a, t in zip(answers, targets)]


class NextTokenAccuracyGrader(BaseGrader):
    """
    Computes the accuracy of the next token prediction.
    """

    def grade(self, answers: list[list[int]], targets: list[list[int]]) -> list:
        return [np.mean([a[i] == t[i] for i in range(len(a))]) for a, t in zip(answers, targets)]


class PromptLengthGrader(BaseGrader):
    """
    Prompts the model with more and more tokens from the proof until it predicts the target.
    """

    def __init__(self, model):
        self.model = model

    def grade(self, answers: list[str], targets: list[str]) -> list:
        pass


class AIPredictionGrader(BaseGrader):
    """
    Prompts the model with the proof and returns the model's prediction.
    """

    def __init__(self, model):
        self.model = model

    def grade(self, answers: list[str], targets: list[str]) -> list:
        pass

# def exact_match(pred, target):
#     """
#     Computes the exact match accuracy of the proof.
#     """
#     return torch.sum(pred == target).item() == len(pred)
#
#
# def next_token_accuracy(pred, target):
#     """
#     Computes the accuracy of the next token prediction.
#     """
#     return torch.sum(pred == target).item() / len(pred)
#
#
# def ai_prediction(validation_model, pred, target):
#     """
#     Prompts the model with the proof and returns the model's prediction.
#     """
#     pass
#
#
# def prompt_length(model, prompt, target):
#     """
#     Prompts the model with more and more tokens from the proof until it predicts the target.
#     """
#     pass
