"""
Loads datasets from huggingface datasets library
Current datasets are
Boxed answers:
 - EleutherAI/hendrycks_math
 - gsm8k

Proofs:
"""

import datasets
import re
from dataclasses import dataclass


@dataclass
class Dataset:
    question: str
    answer: str


def _default_map(data):
    return data


@dataclass
class BoxedDataset:
    boxed: str


@dataclass
class HendrycksDataset(BoxedDataset):
    level: str
    type: str


def _map_hendrycks_data(data):
    return {
        'question': data['problem'],
        'answer': data['problem'],
        'boxed': re.search(r'\\boxed{(.*)}', data['problem']).group(1),
        'level': data['level'],
        'type': data['type']
    }


@dataclass
class GSM8KDataset(BoxedDataset):
    pass


def _map_gsm8k_data(data):
    return {
        'question': data['question'],
        'answer': data['answer'],
        'boxed': re.search(r'\\n#### (.*)', data['question']).group(1)
    }


BOXED_ANSWERS_DATASETS = [
    {
        'name': 'EleutherAI/hendrycks_math',
        'configs': ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                    'prealgebra', 'precalculus'],
        'class': HendrycksDataset,
        'map': _map_hendrycks_data
    }, {
        'name': 'gsm8k',
        'configs': ['main'],
        'class': GSM8KDataset,
        'map': _map_gsm8k_data
    }
]
PROOFS_DATASETS = []


def load_datasets(datasets_list):
    """
    Loads datasets from huggingface datasets library
    """
    return [datasets.load_dataset(dataset['name'], config).map(dataset['map']) for dataset in datasets_list
            for config in dataset['configs']]
