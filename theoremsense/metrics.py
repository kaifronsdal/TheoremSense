"""
Contains implementations of various proxy metrics for measuring the correctness of a theorem proof.
Currently implemented:
To implement:
    - Exact Match
    - Next-Token Prediction Accuracy
    - AI prediction
    - (boxed answer) Prompt Length
"""

import torch


def exact_match(pred, target):
    """
    Computes the exact match accuracy of the proof.
    """
    return torch.sum(pred == target).item() == len(pred)


def next_token_accuracy(pred, target):
    """
    Computes the accuracy of the next token prediction.
    """
    return torch.sum(pred == target).item() / len(pred)


def ai_prediction(validation_model, pred, target):
    """
    Prompts the model with the proof and returns the model's prediction.
    """
    pass


def prompt_length(model, prompt, target):
    """
    Prompts the model with more and more tokens from the proof until it predicts the target.
    """
    pass
