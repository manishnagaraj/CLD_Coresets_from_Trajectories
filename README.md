# Coresets from Trajectories: Selecting Data via Correlation of Loss Differences 

![Alt Text](https://github.com/manishnagaraj/CLD_Coresets_from_Trajectories/blob/main/Visuals/Overview_figure_bar_plot.png)
This repository contains the source code associated with [Coresets from Trajectories: Selecting Data via Correlation of Loss Differences, TMLR 2025](https://openreview.net/forum?id=QY0pbZTWJ9). This code has most recently been tested with Python 3.12 and Pytorch 2.3.1

## Introduction

Deep learning models achieve state-of-the-art performance across domains but face scalability challenges in real-time or resource-constrained scenarios. 
To address this, we propose _Correlation of Loss Differences_ ($\texttt{CLD}$), a simple and scalable metric for coreset selection that identifies the most impactful training samples by measuring their alignment with the loss trajectories of a held-out validation set.  
$\texttt{CLD}$ is highly efficient, requiring only per-sample loss values computed at training checkpoints, and avoiding the costly gradient and curvature computations used in many existing subset selection methods. 
We develop a general theoretical framework that establishes convergence guarantees for $\texttt{CLD}$-based coresets, demonstrating that the convergence error is upper-bounded by the alignment of the selected samples and the representativeness of the validation set. 
On CIFAR-100 and ImageNet-1k, $\texttt{CLD}$-based coresets typically outperform or closely match state-of-the-art methods across subset sizes, and remain within 1\% of more computationally expensive baselines even when not leading.
$\texttt{CLD}$ transfers effectively across architectures (ResNet, VGG, DenseNet), enabling proxy-to-target selection with $<1\%$ degradation. 
Moreover, $\texttt{CLD}$ is stable when using only early checkpoints, incurring negligible accuracy loss. 
Finally, $\texttt{CLD}$ exhibits inherent bias reduction via per-class validation alignment, obviating the need for additional stratified sampling. 
Together, these properties make $\texttt{CLD}$ a principled, efficient, stable, and transferable tool for scalable dataset optimization.


## Installation

Clone this repository using: ```git clone https://github.com/manishnagaraj/CDL_correlation_of_loss_differences.git```

Create a conda environment using the environment.yml file: ```conda env create -f environment.yml```

Activate conda environment ```conda activate cld```

You can also manually create an environment ensuring the following packages are installed
##### Requirements
- python (3.12)
- pytorch (2.3)
- fire 
- numpy
- pandas
- torchvision
- tqdm

## Running the Code

### 1) Collect per-example losses (train + val)

**Default behavior (post-epoch eval on train):**

```bash
python get_loss_values.py --data_path <DATA_DIR> \
  --dataset CIFAR100 --model_arch resnet18 
```

### 2) Compute and store CLD

```bash
python Compute_CLD_scores.py --loss_path <PATH_TO_/Scores/..._losses.pickle>
```

### 3) Train on coresets

```bash
python train_on_coresets.py --score_path <PATH_TO_CDL_PICKLE> --samples_per_class <k>
```

### Defaults (current code)

```python
# Get_loss_values.py defaults (loss collection)
data_path: str = './Data',
dataset: str = 'CIFAR100',
model_arch: str = 'resnet18',
workers: int = 4,
epochs: int = 164,
start_epoch: int = 0,
batch_size: int = 128,
test_batch_size: int = 256,
val_split_ratio: float = 0.1,
learning_rate: float = 0.1,
momentum: float = 0.9,
weight_decay: float = 5e-4,
disable_nesterov: bool = False,
schedule: List[int] = [81, 121],
gamma: float = 0.1,
checkpoint_path: str = './Data/Checkpoint',
logpath: str = './Logs',
resume_path: str = '',
manual_seed: int = 1234,
evaluate_only: bool = False
```

## Running Baselines 

For baselines we utilized and followed the [Deepcore repository](https://github.com/PatrickZH/DeepCore)


## Citations

If you find this code useful in your research, please consider citing our main paper: [Nagaraj, Manish, Deepak Ravikumar, and Kaushik Roy. "Coresets from Trajectories: Selecting Data via Correlation of Loss Differences." Transactions on Machine Learning Research (2025).](https://openreview.net/forum?id=QY0pbZTWJ9)
```
@article{
nagaraj2025coresets,
title={Coresets from Trajectories: Selecting Data via Correlation of Loss Differences},
author={Manish Nagaraj and Deepak Ravikumar and Kaushik Roy},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=QY0pbZTWJ9},
note={}
}
```


## Authors

[Manish Nagaraj](https://manishnagaraj.github.io/), [Deepak Ravikumar](https://deepaktatachar.github.io/), [Kaushik Roy](https://engineering.purdue.edu/NRL)

All authors are with Purdue University, West Lafayette, IN, USA

## Acknowledgement

*This work was supported in part by the Center for the Co-Design of Cognitive Systems (CoCoSys), a DARPA-sponsored JUMP 2.0 center, the Semiconductor Research Corporation (SRC), the National Science Foundation, and Collins Aerospace. We are also thankful to [Efstathia Soufleri](https://efstathia-soufleri.github.io/), [Utkarsh Saxena](https://github.com/UtkarshSaxena1), [Amitangshu Mukherjee](https://github.com/Amitangshu1013), and [Sakshi Choudhary](https://github.com/Sakshi09Ch) for their helpful discussions and feedback.*
