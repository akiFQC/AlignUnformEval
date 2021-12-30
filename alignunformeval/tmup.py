from typing import Callable, Dict, Optional

import numpy as np
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
        encode_fn: Callable[[str], np.ndarray],
        path: Optional[str] = None,
    ) -> None:
        if path is None:
            self._download(self.URL)
            df = pd.read_table(self._get_cache_path())
        else:
            df = pd.read_table(path)

        index_para = df[df[self.COL_LABEL] == self.LABEL_PARA].index
        texts_a = df.loc[index_para, self.COL_SENT_A].to_list()
        texts_a = [text.replace(" ", "") for text in texts_a]
        texts_b = df.loc[index_para, self.COL_SENT_B].to_list()
        texts_b = [text.replace(" ", "") for text in texts_b]
        texts_total = df[self.COL_SENT_A].tolist() + df[self.COL_SENT_B].tolist()
        super().__init__(encode_fn, texts_a, texts_b, texts_total)
