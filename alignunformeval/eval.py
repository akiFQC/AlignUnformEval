import random
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def eval_alignment(encode_fn: Callable[[str], np.ndarray], texts_a: list[str], texts_b: list[str]):
    vec_a = np.array([list(encode_fn(text)) for text in texts_a])
    vec_b = np.array([list(encode_fn(text)) for text in texts_b])
    return np.mean(np.linalg.norm(np.abs(vec_a - vec_b), axis=1, ord=2))


def eval_uniform(encode_fn: Callable[[str], np.ndarray], texts_a: list[str]):
    vec = np.array([list(encode_fn(text)) for text in texts_a])
    dists = pdist(vec, "euclidean")
    return np.log(np.mean(np.exp(-2 * dists)))


class BaseEval(object):
    def __init__(
        self,
        encode_fn: Callable[[str], np.ndarray],
        texts_a: list[str],
        texts_b: list[str],
        texts_total: list[str],
    ) -> None:
        self.encode_fn = encode_fn
        self.text_a = texts_a
        self.text_b = texts_b
        self.text_total = texts_total

    def eval_alignment(self) -> float:
        return eval_alignment(self.encode_fn, self.text_a, self.text_b)

    def eval_uniform(self) -> float:
        return eval_uniform(self.encode_fn, self.text_total)

    def eval_summary(self) -> Dict[str, float]:
        return {"alignment": self.eval_alignment(), "uniform": self.eval_uniform()}
