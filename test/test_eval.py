import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from encode import encoder

from alignunformeval import eval

TEXTS = [
    "my name is tom",
    "sukiyaki",
    "I like sukiyaki",
    "sukiyaki party",
    "my name is ye",
]


def test_align_same():
    align = eval.eval_alignment(encoder, texts_a=TEXTS, texts_b=TEXTS)
    print("align = {0}".format(align))
    assert align == 0.0


def test_align_insame():
    align = eval.eval_alignment(encoder, texts_a=TEXTS, texts_b=TEXTS[::-1])
    print("align = {0}".format(align))


def test_uniform():
    unifo = eval.eval_uniform(encoder, texts_a=TEXTS)
    print("uniform = {0}".format(unifo))
