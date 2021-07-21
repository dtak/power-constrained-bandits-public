# Power Constrained Bandits

Public repo containing code for simulations, bandit algorithms, wrapper algorithms for paper "Power Constrained Bandits"

# Overview

This repo is based on the following academic publication:


* Paper PDF: https://arxiv.org/abs/2004.06230v3

### Contents

* [environments.py](https://github.com/dtak/power-constrained-bandits-public/blob/main/environments.py)
** Provided environments for simulations including SCB, ASCB, HeartSteps
  
* [algorithms.py]
** Provided algorithms used as baselines (ACTS, BOSE, LinUCB)
 
* [wrappers_algs.py](https://github.com/dtak/power-constrained-bandits-public/)
** wrappers

# Installation

* Step 1: Clone this repo

git clone https://github.com/dtak/power-constrained-bandits-public/

* Step 2: Setup a fresh conda enviroment with all required Python packages

bash [`$PC_REPO_DIR/scripts/install/create_conda_env.sh`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/scripts/install/create_conda_env.sh)

* Step 3: Compile Cython code for per-document inference (makes things very fast)

`cd $PC_REPO_DIR/`

python [`setup.py`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/setup.py) `build_ext --inplace`

* Step 4: (Optional) Install tensorflow

bash [`$PC_REPO_DIR/scripts/install/install_tensorflow_linux.sh`](https://github.com/dtak/prediction-constrained-topic-models/tree/master/scripts/install/install_tensorflow_linux.sh)

# Configuration

Set up your environment variables!

First, make a shortcut variable to the location of this repo, so you can easily reference datasets, etc.

    $ export PC_REPO_DIR=/path/to/prediction_constrained_topic_models/

Second, add this repo to your python path, so you can do "import pc_toolbox"

    $ export PYTHONPATH=$PC_REPO_DIR:$PYTHONPATH


# Examples

## Python script to run simulations

The primary script is train_and_eval_sklearn_binary_classifier.py
```
python run_exp.py\
```




