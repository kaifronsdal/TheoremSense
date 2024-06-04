import os

from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()

GPU_MAPPING = {
    0: "2",
    1: "3",
    2: "4",
    3: "5",
    4: "6",
    5: "7",

    6: "2",
    7: "3",
    8: "4",
    9: "5",
    10: "6",
    11: "7",

    12: "2",
    13: "3",
    14: "4",
}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_MAPPING[args.index]

from pathlib import Path
import pandas as pd
from latex_formater import get_final_answer

save_path = Path('~/GitHub/gold-ai-olympiad/data/MATH/Predictions/').expanduser()


# save_path.mkdir(parents=True, exist_ok=True)


class Metric():
    def __init__(self):
        super().__init__()

    def process(self, results):
        pass

    def __call__(self, results):
        return self.process(results)


class BoxedMatch(Metric):
    def process(self, results):
        results['boxed_pred'] = results['prediction'].apply(get_final_answer)
        results['boxed_true'] = results['boxed']
        results['boxed_match'] = results['boxed_true'] == results['boxed_pred']

        # very slow, probably don't use
        # results['match'] = results.apply(lambda x: is_equiv(x['boxed_true'], x['boxed_pred']), axis=1)

        # set first columns to be ['dataset', 'i', 'model', 'method', 'boxed_true', 'boxed_pred', 'match', ...]
        # cols = ['dataset', 'i', 'model', 'method', 'boxed_true', 'boxed_pred', 'boxed_match']
        # cols.extend([col for col in results.columns if col not in cols])
        return results  # [cols]


# import torch
#
# torch.set_float32_matmul_precision('medium')

# from comet import download_model, load_from_checkpoint
#
#
# class COMET(Metric):
#     def __init__(self, model_name="Unbabel/XCOMET-XL", batch_size=32, gpus=1):
#         super().__init__()
#         self.model_name = model_name
#         self.model_path = download_model(model_name)
#         self.model = load_from_checkpoint(self.model_path)
#
#         self.batch_size = batch_size
#         self.gpus = gpus
#
#     def process(self, results):
#         data = [
#             {
#                 "src": str(row['problem']),
#                 "mt": str(row['prediction']),
#                 "ref": str(row['solution'])
#             } for _, row in results.iterrows()
#         ]
#         model_output = self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus)
#         # results[self.model_name] = model_output
#         return model_output

# comet = COMET()
# print('COMET')
# results = comet(results)
#
# temp_save_path = save_path / 'temp2'
# # ensure exists
# temp_save_path.mkdir(parents=True, exist_ok=True)
# # results.to_json(temp_save_path)
# import pickle
#
# with open(temp_save_path / 'comet.pkl', 'wb') as f:
#     pickle.dump(results, f)

# def __init__(self):
#     super().__init__()
#
# def process(self, results):
#     pass
#
# def __call__(self, results):
#     return self.process(results)


from nltk.tokenize import sent_tokenize

from roscoe.score import (
    SEQ_EMB_MODEL_TYPES,
    Chain,
    Evaluator,
    REASONING_SCORES,
    UNSUPERVISED_SCORES,
    SENT_TRANS,
    SIMSCE
)
from roscoe.utils import (
    print_and_reset_max_gpu_memory,
    save_scores,
    split_gsm8k_gpt3_generations_to_steps,
)


class ReasoningSteps(Chain):
    def __init__(self, line: str, type="regular") -> None:
        self.chain = self.parse_chain(line, type=type)

    def parse_chain(self, chain: str, type: str) -> list[str]:
        """
        Change formatting.

        Returns list of steps in reasoning chain.
        """
        if type == "gsm8k_ref":
            return chain.split("IGNORE THIS. Ground truth here for reference. ")[
                1
            ].split('\n')
        elif type == "gsm8k_hypo":
            return split_gsm8k_gpt3_generations_to_steps(reasoning=chain)
        elif type == "regular":
            return sent_tokenize(chain)
        else:
            raise NotImplementedError(f"{type} chain type is not supported")


from roscoe.score import Evaluator


class ROSCOE(Metric):
    def __init__(self, score_types=REASONING_SCORES, model_type=SIMSCE,
                 transformer_model="facebook/roscoe-512-roberta-base", ppl_model="gpt2-large", discourse_batch=64 * 2,
                 coherence_batch=64 * 2, ppl_batch=64 * 2, grammar_batch=64 * 2):
        super().__init__()
        self.evaluator = Evaluator(
            score_types=score_types,
            model_type=model_type,
            transformer_model=transformer_model,
            ppl_model=ppl_model,
            discourse_batch=discourse_batch,
            coherence_batch=coherence_batch,
            ppl_batch=ppl_batch,
            grammar_batch=grammar_batch,
            hypos=[],
            context=[],
        )

    def process(self, results):
        hypos = [
            ReasoningSteps(str(pred))
            for pred in results['prediction']
        ]
        refs = [
            ReasoningSteps(str(solution))
            for solution in results['solution']
        ]
        context = [
            ReasoningSteps(str(problem))
            for problem in results['problem']
        ]

        self.evaluator.set_hypos(hypos)
        self.evaluator.set_references(refs)
        self.evaluator.set_context(context)
        scores = self.evaluator.evaluate()
        # print(scores)
        # results['ROSCOE'] = scores
        return scores


# comet = COMET()
# print('COMET')
# results = comet(results)
#
# temp_save_path = save_path / 'COMET'
# # ensure exists
# temp_save_path.mkdir(parents=True, exist_ok=True)
# # results.to_json(temp_save_path)
# import pickle
#
# with open(temp_save_path / 'comet2.pkl', 'wb') as f:
#     pickle.dump(results, f)

# because scoring using roscoe takes so long, we will score in batches
# and checkpoint the results
temp_save_path = save_path / 'ROSCOE'
# ensure exists
temp_save_path.mkdir(parents=True, exist_ok=True)
# group by dataset
roscoe = ROSCOE()
import pickle
import numpy as np

# print(f'total len: {len(results.groupby("dataset"))}')

dataset_groups = ['Number_Theory_test', 'Counting_and_Probability_train', 'Algebra_test',
                  'Geometry_test', 'Number_Theory_train', 'Prealgebra_test',
                  'Intermediate_Algebra_train', 'Prealgebra_train', 'Precalculus_train',
                  'Intermediate_Algebra_test', 'Counting_and_Probability_test',
                  'Geometry_train', 'Algebra_train', 'Precalculus_test']

# split amongst 3 runs
num_runs = len(GPU_MAPPING)
i = args.index
datasets_for_this_run = dataset_groups[i::num_runs]
print(f'Processing {datasets_for_this_run}')

# load the results and combine them back into a single dataframe
results = pd.concat([
    pd.read_json(save_path / f)
    for f in save_path.iterdir()
    if f.suffix == '.json' and f.stem in datasets_for_this_run
])

print(f'Loaded {len(results)} results')

# results = results[results['dataset'].isin(datasets_for_this_run)]
results = results.query('method == "autoregressive"')

# for dataset, group in results.groupby('dataset'):
# iterate in reverse order so that we can checkpoint our progress
for dataset, group in reversed(list(results.groupby('dataset'))):
    # sample 300 examples per model
    question_ids = np.random.choice(group['i'].unique(), size=100, replace=False)
    print(f'Number of questions: {len(question_ids)}')
    group = group[group['i'].isin(question_ids)]
    print(f'Processing {dataset} with {len(group)} examples and {len(group["model"].unique())} models')
    # breakpoint()
    output = roscoe(group)
    # group.to_json(temp_save_path / f'{dataset}.json')
    with open(temp_save_path / f'{dataset}.pkl', 'wb') as f:
        pickle.dump(output, f)
