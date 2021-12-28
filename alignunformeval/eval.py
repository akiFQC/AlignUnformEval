import pandas as pd
import numpy as np
from typing import Callable, Dict, Tuple, Callable
import random
from scipy.spatial.distance import pdist


class BaseEval(object):
    def __init__(
        self, encode_fn: Callable, texts_a: list[str], texts_b: list[str]
    ) -> None:
        self.encode_fn = encode_fn
        self.text_a = texts_a
        self.text_b = texts_b
        self.vec_a = np.array([list(self.encode_fn(text)) for text in texts_a])
        self.vec_b = np.array([list(self.encode_fn(text)) for text in texts_b])
    

    def eval_alignment(self):
        return np.mean(np.linalg.norm(np.abs(self.vec_a - self.vec_b), axis=1, ord=2))

    def eval_uniform(self):
        concated = np.concatenate([self.vec_a, self.vec_b], axis=0)
        dists = pdist(concated, 'euclidean')
        return np.log(np.mean(np.exp(-2 * dists)))
        