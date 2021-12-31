import numpy as np
from encode import hash_encode
from pandas.core.algorithms import isin

from alignunformeval import STSBEval


def test_stsb_eval():
    evaluator = STSBEval(hash_encode)
    result = evaluator.eval_summary()
    assert isinstance(result, dict)
    assert "alignment" in result
    assert "uniformity" in result
    assert isinstance(result["alignment"], float)
    assert isinstance(result["uniformity"], float)
    print(f"result={result}")
