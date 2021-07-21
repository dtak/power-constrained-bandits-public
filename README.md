## Power Constrained Bandits

Public repo containing code for simulations, bandit algorithms, wrapper algorithms for paper "Power Constrained Bandits"

## Overview

This repo is based on the following academic publication:


* Paper PDF: https://arxiv.org/abs/2004.06230v3

### Contents

* [environments.py](https://github.com/dtak/power-constrained-bandits-public/blob/main/environments.py)
    *Provided environments for simulations including SCB, [ASCB](https://arxiv.org/abs/1803.04204), [HeartSteps](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4848174/)
  
* [algorithms.py](https://github.com/dtak/power-constrained-bandits-public/blob/main/algorithms.py)
    * Provided algorithms used as baselines including fixed $\pi=0.5$ [ACTS](https://arxiv.org/abs/1711.03596), [BOSE](https://arxiv.org/abs/1803.04204), [LinUCB](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html))
 
* [wrappers_algs.py](https://github.com/dtak/power-constrained-bandits-public/blob/main/wrapper_algs.py)
    * Provided meta-algorithms we developed including probability clipping, data dropping and action flipping

## Library Requirements
python>=3.5, numpy, argparse, scipy

## Examples

### Python script to run simulations
The main script to run power analyses simulations on SCB environment with ACTS algorithm fix $\pi$ and probability-clipping meta-algorithm
```
python run_exp.py -S 1000 #number of simulations\
-env scb # environments (options: scb, ascb, mobile, nonlinear) \
-alg acts # algorithms (options: fix_pi, acts, bose, linucb) \
-wrapper clip # wrapper algorithms (options: none, clip, drop, flip)\
-experiment power # experiment type (options: power)\
```




