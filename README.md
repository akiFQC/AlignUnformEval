# AlignUnformEval
[![PyPI version](https://badge.fury.io/py/alignunformeval.svg)](https://badge.fury.io/py/alignunformeval)
[![pytest](https://github.com/akiFQC/AlignUnformEval/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/akiFQC/AlignUnformEval/actions/workflows/ci.yaml)

This is a python tool to evaluate alignment and uniformity of sentence embedding like [SimCSE paper](https://arxiv.org/pdf/2104.08821.pdf).   

[SimCSE paper](https://arxiv.org/pdf/2104.08821.pdf) explains alignment and uniformity as below:
>  Given a distribution of positive
pairs p_pos, alignment calculates expected distance between embeddings of the paired instances (assuming representations are already normalized): 
<img src="https://latex.codecogs.com/gif.latex?\ell_{\rm align}:=\mathbb{E}_{(x, x^{+})\sim p_{\rm pos}}\left[\| f(x)-f(x^{+}) \|^{2} \right]" />


> On the other hand, uniformity measures how well
the embeddings are uniformly distributed:
<img src="https://latex.codecogs.com/gif.latex?\ell_{\rm uniformity}:=\log \mathbb{E}_{(x, y) \overset{i.i.d.}{\sim}  p_{\rm data}} \left[e ^{ -2\| f(x)-f(x^{+}) \|^{2}}\right]" />
> where p_data denotes the data distribution. 

## Install
by pip
```
pip install alignuniformeval
``` 

by source
```
pip install https://github.com/akiFQC/AlignUnformEval
```


## Usage
You can easily evaluate alignment and uniformity with this library.  
This is a minimal example that evaluate alignment and uniformity of STS Benchmark.
```
from alignunformeval import STSBEval

evaluator = STSBEval(sentence_encoder)
# sentence_encoder is a callable from List[str] to numpy.array. The output numpy.array must be [dimention_of_sentence_vector].
result = evaluator.eval_summary()
# result =  {"alignment": value_of_aligenment, "uniformity": value_of_uniformity}
```
`STSBEval` get callable whose input is `list` of `str` and output is n dimentional `numpy.array`.

## Dataset

### [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)
This dataset (especially, `sts-dev.csv`) was used in [SimCSE paper](https://arxiv.org/pdf/2104.08821.pdf). In the paper, the threshold of similarity score was st at 4.0;  pairs of sentences whose similarity score is higher than 4.0 are used for evaluation of alignment. You can set other threshold as the following example.
```
from alignunformeval import STSBEval
# sentence_encoder : some function List[str] to np.array[dimention_of_sentence_vector]
evaluator = STSBEval(sentence_encoder, threshold=3.0) # set threshold at 3.0
result = evaluator.eval_summary()
```
Please see `test/test_stsb.py` if you want more details.

### [Tokyo Metropolitan University Paraphrase Corpus (TMUP)](https://github.com/tmu-nlp/paraphrase-corpus)
[Tokyo Metropolitan University Paraphrase Corpus (TMUP)](https://github.com/tmu-nlp/paraphrase-corpus) is a Japanese paraphrase dataset.

```
from alignunformeval import TMUPEval
# sentence_encoder : some function List[str] to np.array[dimention_of_sentence_vector]
evaluator = TMUPEval(sentence_encoder)
result = evaluator.eval_summary()
```

## License 
The license of this tool follows each dataset. Please read the documents of datasets you use.

## Reference
1. https://arxiv.org/pdf/2104.08821.pdf 
2. https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark 
3. https://github.com/tmu-nlp/paraphrase-corpus

