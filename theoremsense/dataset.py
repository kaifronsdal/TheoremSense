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
import hashlib


def get_hash(input_string):
    sha_signature = hashlib.sha256(input_string.encode()).hexdigest()
    return sha_signature


class Datapoint(dict):
    def __init__(self, *args, format: str | None = None, **kwargs):
        super(Datapoint, self).__init__(*args, **kwargs)
        self._format = format

    def format(self):
        if self._format is not None:
            return eval(f'f"""{self._format}"""', self)
        return None

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"No such attribute: {attr}")

    def __setattr__(self, attr, value):
        if attr == '_format':
            self.__dict__['_format'] = value  # store _format as a property
        else:
            self[attr] = value

    def __delattr__(self, attr):
        if attr == '_format':
            del self.__dict__['_format']  # delete the _format property
        elif attr in self:
            del self[attr]
        else:
            raise AttributeError(f"No such attribute: {attr}")


def matching_brackets(string, idx, open_char='{', close_char='}'):
    assert string[idx] == open_char
    if idx < len(string):
        opening = [i for i, c in enumerate(string[idx + 1:]) if c == open_char]
        closing = [i for i, c in enumerate(string[idx + 1:]) if c == close_char]
        for i, j in enumerate(closing):
            if i >= len(opening) or j < opening[i]:
                return j + idx + 1
    return -1


def get_boxed_answer(answer: str):
    match = re.search(r'\\boxed{', answer)
    boxed = ''
    if match is not None:
        boxed_start = match.end() - 1
        boxed_end = matching_brackets(answer, boxed_start)
        boxed = answer[boxed_start + 1:boxed_end]
    else:
        match = re.search(r'\\boxed ', answer)
        if match is not None:
            boxed = answer[match.end()]
    return boxed


def get_boxed_answers(answers: list[str]):
    return [get_boxed_answer(answer) for answer in answers]


def _default_map(data):
    return data


def _format_hendrycks_data(dataset):
    def map_data(data):
        answer = data['answer'].replace('\\!', '')
        data['question'] = data['question'].replace('\\!', '')

        data['answer'] = answer
        data['boxed'] = get_boxed_answer(answer)
        data['hash'] = get_hash(data['question'] + data['answer'])
        return data

    dataset = dataset.rename_column('problem', 'question')
    dataset = dataset.rename_column('solution', 'answer')
    dataset = dataset.map(map_data)
    return dataset


def _format_gsm8k_data(dataset):
    def map_data(data):
        data['boxed'] = re.search(r'\n#### (.*)', data['answer']).group(1)
        data['hash'] = get_hash(data['question'] + data['answer'])
        return data

    return dataset.map(map_data)


BOXED_ANSWERS_DATASETS = [
    {
        'name': 'EleutherAI/hendrycks_math',
        'subsets': ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                    'prealgebra', 'precalculus'],
        'format': _format_hendrycks_data
    }, {
        'name': 'gsm8k',
        'subsets': ['main'],
        'format': _format_gsm8k_data
    }
]
PROOFS_DATASETS = []

ALL_DATASETS = BOXED_ANSWERS_DATASETS + PROOFS_DATASETS


def load_dataset(dataset, subset):
    # try:
    #     data = datasets.load_dataset(dataset['name'], subset)
    # except ValueError:  # @TODO: get right exception type
    #     data = datasets.load_dataset(dataset['name'], subset, download_mode='force_redownload',
    #                                  verification_mode='no_checks')
    return {
        'name': f"{dataset['name']}_{subset}",
        'data': dataset['format'](
            datasets.load_dataset(dataset['name'], subset)
            # , download_mode='force_redownload', verification_mode='no_checks')
        )
    }


def load_datasets(datasets_list):
    """
    Loads datasets from huggingface datasets library
    """
    return [
        load_dataset(dataset, subset)
        for dataset in datasets_list
        for subset in dataset['subsets']
    ]


def from_name(name, subset: str | None = None):
    """
    Loads a dataset from its name. If subset is None, loads all subsets of the dataset. Otherwise, loads the specified
    subset.
    """
    if isinstance(name, list):
        datas = []
        for dataset in ALL_DATASETS:
            if dataset['name'] in name:
                data = load_datasets([dataset])
                datas.extend(data)
        return datas

    for dataset in ALL_DATASETS:
        if name == dataset['name']:
            if subset is None:
                return load_datasets([dataset])
            else:
                return load_dataset(dataset, subset)
