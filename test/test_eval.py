import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from alignunformeval import eval

TEXTS = [
    'my name is tom',
    'sukiyaki',
    'I like sukiyaki',
    'sukiyaki party',
    'my name is ye',
    ]

def encoder(text):  
    o_a = ord("a")
    o_z = ord('z')
    ord_list = list(range(o_a, o_z)) + [ord('.'), ord(' '), ord(',')]
    c2i = {o:i for i, o in enumerate(ord_list)}
    arr = np.zeros(shape=len(ord_list)+1)
    for s in text:
        o = ord(s)
        if o in ord_list:
            arr[c2i[o]] += 1
        else:
            arr[-1] += 1
    return arr


def test_init():
    evaler = eval.BaseEval(encoder, texts_a=TEXTS, texts_b=TEXTS)
    print(evaler.vec_a)
    print(type(evaler.vec_a))
    print(evaler.vec_a.shape)
    print(evaler.vec_a.dtype)

def test_align_same():
    evaler = eval.BaseEval(encoder, texts_a=TEXTS, texts_b=TEXTS)
    align = evaler.eval_alignment()
    print('align = {0}'.format(align))
    assert align == 0.0

def test_align_insame():
    evaler = eval.BaseEval(encoder, texts_a=TEXTS, texts_b=TEXTS[::-1])
    align = evaler.eval_alignment()
    print('align = {0}'.format(align))

def test_uniform():
    evaler = eval.BaseEval(encoder, texts_a=TEXTS, texts_b=TEXTS)
    unifo = evaler.eval_uniform()   
    print('uniform = {0}'.format(unifo))