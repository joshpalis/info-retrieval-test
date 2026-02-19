import pytrec_eval
import logging
from typing import List, Dict, Tuple, Any
from math import floor
import numpy as np

logger = logging.getLogger(__name__)


class EvaluateRetrieval:

    def __init__(self, retriever,
                 k_values: List[int] = [1, 3, 5, 10, 100, 1000]):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever

    # def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], **kwargs) -> Dict[
    #     str, Dict[str, float]]:
    #     if not self.retriever:
    #         raise ValueError("Model/Technique has not been provided!")
    #     return self.retriever.search(corpus, queries, self.top_k, **kwargs)

    @staticmethod
    def evaluate(qrels: Dict[str, Dict[str, int]],
                results: Dict[str, Dict[str, float]],
                k_values: List[int],
                ignore_identical_ids: bool = True) -> Tuple[
        Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

        if ignore_identical_ids:
            print('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        # Changed from logging.info to print
        for eval_dict, name in [(ndcg, "NDCG"), (_map, "MAP"), (recall, "Recall"), (precision, "Precision")]:
            print(f"\n=== {name} ===")
            for k, v in eval_dict.items():
                print(f"{k}: {v:.4f}")

        return ndcg, _map, recall, precision

    def _pxx(self, values: List[Any], p: float):
        """Calculates the pXX statistics for a given list.

        Args:
            values: List of values.
            p: Percentile (between 0 and 1).

        Returns:
            The corresponding pXX metric.
        """
        lowest_percentile = 1 / len(values)
        highest_percentile = (len(values) - 1) / len(values)

        # return -1 if p is out of range or if the list doesn't have enough elements
        # to support the specified percentile
        if p < 0 or p > 1:
            return -1.0
        elif p < lowest_percentile or p > highest_percentile:
            if p == 1.0 and len(values) > 1:
                return float(values[len(values) - 1])
            return -1.0
        else:
            return float(values[floor(len(values) * p)])

    def evaluate_time(self, took_time_all_measures):
        p50 = []
        p90 = []
        p99 = []

        for took_time in took_time_all_measures:
            times = list(took_time.values())

            p50.append(np.percentile(np.array(times), 50))
            p90.append(np.percentile(np.array(times), 90))
            p99.append(np.percentile(np.array(times), 99))

        print('p50: ' + str(np.average(p50)))
        print('p90: ' + str(np.average(p90)))
        print('p99: ' + str(np.average(p99)))

        return p50, p90, p99
