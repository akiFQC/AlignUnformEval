import os
import tarfile
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from genericpath import exists
from pandas.io.parsers import read_table

from .eval import BaseEval


class STSBEval(BaseEval):

    URL = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
    FILENAME = os.path.join("stsbenchmark", "sts-dev.csv")
    COL_SCORE = "score"
    COL_SENT_A = "sentence1"
    COL_SENT_B = "sentence2"
    COLUMNS = ["genre", "filename", "year", "id", "score", "sentence1", "sentence2"]

    def __init__(
        self,
        encode_fn: Callable[[str], np.ndarray],
        path: Optional[str] = None,
        threshold: float = 4.0,
    ) -> None:
        self.score_thresh = threshold
        if path is None:
            self.path_dev = os.path.join(self._get_cache_path(), self.FILENAME)
            self.path_targz = str(self._get_cache_path()) + ".tar.gz"
            if not os.path.exists(self.path_dev):
                self._download(self.URL, path=self.path_targz)
                with tarfile.open(self.path_targz, "r:*") as tar:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(tar, str(self._get_cache_path()))
            df = pd.read_csv(
                self.path_dev,
                header=None,
                encoding="utf=8",
                sep="\t",
                names=self.COLUMNS,
            )
        else:
            df = pd.read_table(path, header=None, encoding="utf=8", sep="\t")
        index_para = df[df.loc[:, self.COL_SCORE] > self.score_thresh].index
        texts_a = df.loc[index_para, self.COL_SENT_A].to_list()
        texts_b = df.loc[index_para, self.COL_SENT_B].to_list()
        texts_total = (
            df.loc[:, self.COL_SENT_A].tolist() + df.loc[:, self.COL_SENT_B].tolist()
        )
        # print('len=',len(texts_total))
        super().__init__(encode_fn, texts_a, texts_b, texts_total)
