from typing import Callable, Dict

import pandas as pd

from .eval import BaseEval


class TMUPEval(BaseEval):

    URL = "https://raw.githubusercontent.com/tmu-nlp/paraphrase-corpus/master/tmup.tsv"
    COL_LABEL = "label"
    LABEL_PARA = 1
    COL_SENT_A = "sentence_A_ja"
    COL_SENT_B = "sentence_B_ja"

    def __init__(
        self,
        encode_fn: Callable,
    ) -> None:
        df = pd.read_table(self.URL)
        print("df", df.shape)
        print("df", df.columns)
        print("df", df.tail(2))
        index_para = df[df[self.COL_LABEL] == self.LABEL_PARA].index
        texts_a = df.loc[index_para, self.COL_SENT_A].to_list()
        texts_a = [text.replace(" ", '') for text in texts_a]
        texts_b = df.loc[index_para, self.COL_SENT_B].to_list()
        texts_b = [text.replace(" ", '') for text in texts_b]
        texts_total = texts_a + texts_b
        super().__init__(encode_fn, texts_a, texts_b, texts_total)
