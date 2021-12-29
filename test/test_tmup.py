from alignunformeval import TMUPEval
import numpy as np



def hash_encode(text: str):
    hashed = hash(text)
    out = []
    dim = 36
    for i in range(1, dim+1):
        out.append( hashed// i +  0.12 * hashed % i )
    out=np.array(out)
    return out / np.max(np.abs(out))

def test_tumpeval():
    evaluator = TMUPEval(hash_encode)
    result = evaluator.eval_summary()
    assert isinstance(result, dict)
    assert 'alignment' in result
    assert 'uniform' in result
    print(f'result={result}')