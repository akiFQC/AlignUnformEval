import os
import random
import sys
import urllib.request
from os.path import expanduser
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from genericpath import exists
from scipy.spatial.distance import pdist
from scipy.special import logsumexp


def eval_alignment(
    encode_fn: Callable[[str], np.ndarray], texts_a: list[str], texts_b: list[str]
):
    vec_a = np.array([list(encode_fn(text)) for text in texts_a])
    vec_b = np.array([list(encode_fn(text)) for text in texts_b])
    return float(
        np.mean(
            np.linalg.norm(
                np.abs(vec_a.astype(np.float128) - vec_b.astype(np.float128)),
                axis=1,
                ord=2,
            )
        )
    )


def eval_uniform(encode_fn: Callable[[str], np.ndarray], texts_a: list[str]):
    vec = np.array([list(encode_fn(text)) for text in texts_a])
    # print('vec', vec.max(), vec.mean(), vec.min())
    dists = pdist(vec, "euclidean")
    n = len(dists)
    return float(logsumexp(-2 * dists) - np.log(n))


class BaseEval(object):

    LIB_ROOT_PATH = os.path.abspath(
        os.path.join(expanduser("~"), ".cache", "align_uniform_eval")
    )

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
        return {"alignment": self.eval_alignment(), "uniformity": self.eval_uniform()}

    def _get_cache_path(self):
        return os.path.abspath(
            os.path.join(
                self.LIB_ROOT_PATH,
                f"{str(abs(hash(sys.version)))}_{str(abs(hash(str(self.__class__.__name__))))}",
            )
        )

    def _download(self, url, path=None):
        if path is None:
            path = self._get_cache_path()
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
