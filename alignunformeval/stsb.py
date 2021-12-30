from typing import Callable, Dict

import pandas as pd

from .eval import BaseEval


class TMUPEval(BaseEval):

    COL_LABEL = "label"
    LABEL_PARA = 1
    COL_SENT_A = "sentence_A_ja"
    COL_SENT_B = "sentence_B_ja"

    def __init__(
        self,
        encode_fn: Callable,
        path = None,
    ) -> None:
        if path is None:
            df = pd.read_table(self.URL)
        else:
            df = pd.read_table(path)
        print("df", df.shape)
        print("df", df.columns)
        print("df", df.tail(2))
        index_para = df[df[self.COL_LABEL] == self.LABEL_PARA].index
        texts_a = df.loc[index_para, self.COL_SENT_A].to_list()
        texts_a = [text.replace(" ", '') for text in texts_a]
        texts_b = df.loc[index_para, self.COL_SENT_B].to_list()
        texts_b = [text.replace(" ", '') for text in texts_b]
        texts_total = df[self.COL_SENT_A].tolist() + df[self.COL_SENT_B].tolist()
        super().__init__(encode_fn, texts_a, texts_b, texts_total)
