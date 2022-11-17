
# cvar-opt

[comment]: #  [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)

[comment]: #  [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

Evaluate the Time To Solution (TTS) [1] for Variational Quantum Eigensolver (VQE) using CVaR [2].

## How to run

<details>
<summary><b>Install dependencies</b></summary>

```bash
# clone project
git clone https://github.com/gscriva/cvar-opt
cd cvar-opt

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```
</details>

Run an optimization job with VQE and CVaR for an Ising ferromagnetic 1D problem 
```bash
python run.py --qubits 6 -v
```

## Refs.

1. Albash, T. & Lidar, D. A. Demonstration of a Scaling Advantage for a Quantum Annealer over Simulated Annealing. [Phys. Rev. X 8, 031016](https://doi.org/10.1103/PhysRevX.8.031016) (2018).
2. Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I. & Woerner, S. Improving Variational Quantum Optimization using CVaR. [Quantum 4, 256](https://doi.org/10.22331/q-2020-04-20-256) (2020).

