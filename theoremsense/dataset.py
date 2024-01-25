"""
Loads datasets from huggingface datasets library
Current datasets are
Boxed answers:
 - EleutherAI/hendrycks_math
 - gsm8k

Proofs:
"""

import datasets

BOXED_ANSWERS_DATASETS = [
    {
        'name': 'EleutherAI/hendrycks_math',
        'configs': ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                    'prealgebra', 'precalculus']
    }, {
        'name': 'gsm8k',
        'configs': ['main']
    }
]
PROOFS_DATASETS = []


def load_datasets(datasets_list):
    """
    Loads datasets from huggingface datasets library
    """
    return [datasets.load_dataset(dataset['name'], config) for dataset in datasets_list
            for config in dataset['configs']]
