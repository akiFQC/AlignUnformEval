# AlignUnformEval
This is a python tool to evaluate alignment and uniformity of sentence embedding like [SimCSE paper](https://arxiv.org/pdf/2104.08821.pdf).   

[SimCSE paper](https://arxiv.org/pdf/2104.08821.pdf) explains about alignment and uniformity as below:
>  Given a distribution of positive
pairs ppos, alignment calculates expected distance
between embeddings of the paired instances (assuming representations are already normalized): 
<img src="https://latex.codecogs.com/gif.latex?\ell_{\rm align}:=\mathbb{E}_{(x, x^{+})\sim p_{\rm pos}}\left[\| f(x)-f(x^{+}) \|^{2} \right]" />


> On the other hand, uniformity measures how well
the embeddings are uniformly distributed:
<img src="https://latex.codecogs.com/gif.latex?\ell_{\rm uniform}:=\log \mathbb{E}_{(x, y) \overset{i.i.d.}{\sim}  p_{\rm data}} \left[e ^{ -2\| f(x)-f(x^{+}) \|^{2}}\right]" />
> where pdata denotes the data distribution. 

## Install
by pip
```
pip install AlignUniformEval
``` 


by source
```
pip istall https://github.com/akiFQC/AlignUnformEval
```


## Usage

## Dataset

### Tokyo Metropolitan University Paraphrase Corpus (TMUP)
[TMUP](https://github.com/tmu-nlp/paraphrase-corpus) is a Japanese paraphrase dataset.




## License 
License of this repositry follows each datasets. Please, read the document of datasets you use.

## Reference
1. https://arxiv.org/pdf/2104.08821.pdf  
2. https://github.com/tmu-nlp/paraphrase-corpus

