import numpy as np
from encode import hash_encode

from alignunformeval import TMUPEval


def test_tumpeval():
    evaluator = TMUPEval(hash_encode)
    result = evaluator.eval_summary()
    assert isinstance(result, dict)
    assert "alignment" in result
    assert "uniformity" in result
    assert isinstance(result["alignment"], float)
    assert isinstance(result["uniformity"], float)
    print(f"result={result}")
