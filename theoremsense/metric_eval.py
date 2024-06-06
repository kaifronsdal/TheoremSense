import os

from argparse import ArgumentParser

parser = ArgumentParser()
# parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()

# GPU_MAPPING = {
#     0: "2",
#     1: "3",
#     2: "4",
#     3: "5",
#     4: "6",
#     5: "7",
#
#     6: "2",
#     7: "3",
#     8: "4",
#     9: "5",
#     10: "6",
#     11: "7",
#
#     12: "2",
#     13: "3",
#     14: "4",
# }

GPU_MAPPING = {
    0: "2",
    1: "3",
    2: "5",
}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU_MAPPING[args.index]

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


import torch

torch.set_float32_matmul_precision('medium')

from comet import download_model, load_from_checkpoint


class COMET(Metric):
    def __init__(self, model_name="Unbabel/XCOMET-XL", batch_size=32, gpus=1):
        super().__init__()
        self.model_name = model_name
        self.model_path = download_model(model_name)
        self.model = load_from_checkpoint(self.model_path)

        self.batch_size = batch_size
        self.gpus = gpus

    def process(self, results):
        data = [
            {
                "src": str(row['problem']),
                "mt": str(row['prediction']),
                "ref": str(row['solution'])
            } for _, row in results.iterrows()
        ]
        model_output = self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus)
        # results[self.model_name] = model_output
        return model_output


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

    EMB_MODEL_SCORES,
    NLI_MODEL_SCORES,
    LANGUAGE_MODEL_SCORES,
    GRAMMAR_MODEL_SCORES,

    FAITHFUL_WORD,
    REPETITION_WORD,

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

EMB_MODEL_SCORES_FILTERED = [e for e in EMB_MODEL_SCORES if e not in [FAITHFUL_WORD, REPETITION_WORD]]

Override = 0

USED_SCORES = {
    0: EMB_MODEL_SCORES_FILTERED,
    1: NLI_MODEL_SCORES,
    2: LANGUAGE_MODEL_SCORES,
    3: GRAMMAR_MODEL_SCORES
}[Override]

USED_SCORES_NAME = {
    0: 'EMB_MODEL_SCORES',
    1: 'NLI_MODEL_SCORES',
    2: 'LANGUAGE_MODEL_SCORES',
    3: 'GRAMMAR_MODEL_SCORES'
}[Override]

print(USED_SCORES)


class ROSCOE(Metric):
    def __init__(self, score_types=USED_SCORES, model_type=SIMSCE,
                 transformer_model="facebook/roscoe-512-roberta-base", ppl_model="gpt2-large",
                 discourse_batch=64 * 2 * 4,
                 coherence_batch=64 * 2 * 4, ppl_batch=64, grammar_batch=64 * 2):
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
        # scores is a dictionary of lists
        try:
            for key, value in scores.items():
                results[key] = value
            return results
        except Exception as e:
            print(e)
            breakpoint()
            return scores


results = pd.concat([
    pd.read_json(save_path / f)
    for f in save_path.iterdir()
    if f.suffix == '.json'
])

results = results.query('method == "autoregressive"')

comet = COMET()
print('COMET')
output = comet(results)

try:
    results['COMET'] = output['score']
except Exception as e:
    print(e)
    breakpoint()

temp_save_path = save_path / 'COMET'
# ensure exists
temp_save_path.mkdir(parents=True, exist_ok=True)
# results.to_json(temp_save_path)
import pickle
import json

try:
    with open(temp_save_path / 'comet_final2.json', 'w') as f:
        json.dump(output, f)
except Exception as e:
    print(f'Error saving')
    print(e)
    breakpoint()

# with open(temp_save_path / 'comet_final2.pkl', 'wb') as f:
#     pickle.dump(output, f)

# # because scoring using roscoe takes so long, we will score in batches
# # and checkpoint the results
# temp_save_path = save_path / 'ROSCOE'
# # ensure exists
# temp_save_path.mkdir(parents=True, exist_ok=True)
# # group by dataset
# roscoe = ROSCOE(transformer_model="princeton-nlp/sup-simcse-roberta-base")
# import pickle
# import numpy as np
# import json
#
# # print(f'total len: {len(results.groupby("dataset"))}')
#
# dataset_groups = ['Number_Theory_test', 'Counting_and_Probability_train', 'Algebra_test',
#                   'Geometry_test', 'Number_Theory_train', 'Prealgebra_test',
#                   'Intermediate_Algebra_train', 'Prealgebra_train', 'Precalculus_train',
#                   'Intermediate_Algebra_test', 'Counting_and_Probability_test',
#                   'Geometry_train', 'Algebra_train', 'Precalculus_test']
#
# # split amongst 3 runs
# num_runs = len(GPU_MAPPING)
# i = args.index
# datasets_for_this_run = dataset_groups[i::num_runs]
# print(f'Processing {datasets_for_this_run}')
#
# # load the results and combine them back into a single dataframe
# results = pd.concat([
#     pd.read_json(save_path / f)
#     for f in save_path.iterdir()
#     if f.suffix == '.json'  # and f.stem in datasets_for_this_run
# ])
#
# print(f'Loaded {len(results)} results')
#
# # results = results[results['dataset'].isin(datasets_for_this_run)]
# results = results.query('method == "autoregressive"')
# #
# # import cProfile
# # import pstats
# #
# # profiler = cProfile.Profile()
# # profiler.enable()
#
# for dataset, group in list(results.groupby('dataset')):
#     # sample 300 examples per model
#     # np.random.seed(42)
#     # question_ids = np.random.choice(group['i'].unique(), size=100, replace=False)
#     # # breakpoint()
#     # print(f'Number of questions: {len(question_ids)}')
#     # group = group[group['i'].isin(question_ids)]
#     print(f'Processing {dataset} with {len(group)} examples and {len(group["model"].unique())} models')
#     # breakpoint()
#     output = roscoe(group.copy())
#
#     # group.to_json(temp_save_path / f'{dataset}.json')
#     # try:
#     #     output.to_json(temp_save_path / f'{dataset}.json')
#     # except Exception as e:
#     #     print(f'Error saving {dataset}')
#     #     print(e)
#     #     breakpoint()
#     try:
#         with open(temp_save_path / f'SUPSIMCE_MODEL_{USED_SCORES_NAME}_{dataset}.json', 'w') as f:
#             json.dump(output.to_dict('records'), f)
#     except Exception as e:
#         print(f'Error saving {dataset}')
#         print(e)
#         breakpoint()
#
# # profiler.disable()
# #
# # # save the profile to a file
# # profiler.dump_stats('profile.prof')
