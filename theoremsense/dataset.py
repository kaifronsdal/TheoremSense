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

def matching_brackets(string, idx, open_char='{', close_char='}'):
    assert string[idx] == open_char
    if idx < len(string):
        opening = [i for i, c in enumerate(string[idx + 1 :]) if c == open_char]
        closing = [i for i, c in enumerate(string[idx + 1 :]) if c == close_char]
        for i, j in enumerate(closing):
            if i >= len(opening) or j < opening[i]:
                return j + idx + 1
    return -1


def _format_hendrycks_data(dataset):
    def map_data(data):
        answer = data['solution'].replace('\\!', '')
        data['solution'] = answer
        match = re.search(r'\\boxed{', answer)
        if match is not None:
            boxed_start = match.end() - 1
            boxed_end = matching_brackets(answer, boxed_start)
            boxed = answer[boxed_start+1:boxed_end]
        else:
            match = re.search(r'\\boxed ', answer)
            boxed = answer[match.end()]
        data['boxed'] = boxed
        return data

    dataset = dataset.map(map_data)
    dataset = dataset.rename_column('problem', 'question')
    dataset = dataset.rename_column('solution', 'answer')
    return dataset


@dataclass
class GSM8KDataset(BoxedDataset):
    pass


def _format_gsm8k_data(dataset):
    def map_data(data):
        data['boxed'] = re.search(r'\n#### (.*)', data['answer']).group(1)
        return data
    return dataset.map(map_data)


BOXED_ANSWERS_DATASETS = [
    {
        'name': 'EleutherAI/hendrycks_math',
        'configs': ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                    'prealgebra', 'precalculus'],
        'class': HendrycksDataset,
        'format': _format_hendrycks_data
    }, {
        'name': 'gsm8k',
        'configs': ['main'],
        'class': GSM8KDataset,
        'format': _format_gsm8k_data
    }
]
PROOFS_DATASETS = []


def load_datasets(datasets_list):
    """
    Loads datasets from huggingface datasets library
    """
    return [
        dataset['format'](
            datasets.load_dataset(dataset['name'], config, download_mode='force_redownload')
        )
        for dataset in datasets_list
        for config in dataset['configs']
    ]
