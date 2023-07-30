<div align="center">

# Accelerating equilibrium spin-glass simulations using quantum data and deep learning

[comment]: <![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)>
[comment]: <[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)>

</div>

Quantum enhanced optimization of classical cost functions is a central theme of quantum computing due to its high potential value in science and technology.
The variational quantum eigensolver (VQE) and the quantum approximate optimization algorithm (QAOA) are popular variational approaches that are considered the most viable solutions in the noisy-intermediate scale quantum (NISQ) era.
Here, we study the scaling of the quantum resources, defined as the required number of circuit repetitions, to reach a fixed success probability as the problem size increases, focusing on the role played by measurement shot noise, which is unavoidable in realistic implementations.
Simple and reproducible problem instances are addressed, namely, the ferromagnetic and disordered Ising chains. 
Our results show that: 
1. VQE with the standard heuristic ansatz scales comparably to direct brute-force search when energy-based optimizers are employed. The performance improves at most quadratically using a gradient-based optimizer.
2. When the parameters are optimized from random guesses, also the scaling of QAOA implies problematically long absolute runtimes for large problem sizes.
3. QAOA becomes practical when supplemented with a physically-inspired initialization of the parameters.
Our results suggest that hybrid quantum-classical algorithms should possibly avoid a brute force classical outer loop, but focus on smart parameters initialization.

<br>

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

# install requirements
pip install -r requirements.txt
```
</details>

Run an optimization job with VQE and CVaR for an Ising ferromagnetic 1D problem with default parameters
```bash
python run.py --qubits 6 -v
```

For the others parameters, take a look to the help
```bash
python run.py --help
```

<br>


## References

1. Albash, T. & Lidar, D. A. Demonstration of a Scaling Advantage for a Quantum Annealer over Simulated Annealing. [Phys. Rev. X 8, 031016](https://doi.org/10.1103/PhysRevX.8.031016) (2018).
2. Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I. & Woerner, S. Improving Variational Quantum Optimization using CVaR. [Quantum 4, 256](https://doi.org/10.22331/q-2020-04-20-256) (2020).

<br>
