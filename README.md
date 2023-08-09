<div align="center">

# Challenges of variational quantum optimization with measurement shot noise
Giuseppe Scriva, Nikita Astrakhantsev, Sebastiano Pilati, and Guglielmo Mazzola

-----

[![Paper](http://img.shields.io/badge/Paper-arXiv%202308.00044-B31B1B.svg)](https://arxiv.org/abs/2308.00044)
[![Data](https://zenodo.org/badge/DOI/10.5281/zenodo.8223528.svg)](https://doi.org/10.5281/zenodo.8223528)

</div>

## Abstract

Quantum enhanced optimization of classical cost functions is a central theme of quantum computing due to its high potential value in science and technology.
The variational quantum eigensolver (VQE) and the quantum approximate optimization algorithm (QAOA) are popular variational approaches that are considered the most viable solutions in the noisy-intermediate scale quantum (NISQ) era.
Here, we study the scaling of the quantum resources, defined as the required number of circuit repetitions, to reach a fixed success probability as the problem size increases, focusing on the role played by measurement shot noise, which is unavoidable in realistic implementations.
Simple and reproducible problem instances are addressed, namely, the ferromagnetic and disordered Ising chains. 
Our results show that: 
1. VQE with the standard heuristic ansatz scales comparably to direct brute-force search when energy-based optimizers are employed. The performance improves at most quadratically using a gradient-based optimizer;
2. When the parameters are optimized from random guesses, also the scaling of QAOA implies problematically long absolute runtimes for large problem sizes;
3. QAOA becomes practical when supplemented with a physically-inspired initialization of the parameters.

Our results suggest that hybrid quantum-classical algorithms should possibly avoid a brute force classical outer loop, but focus on smart parameters initialization.

<br>

## Going deeper

[In our article](https://arxiv.org/abs/2308.00044), we consider optimization problems with VQE and QAOA (eventually enhanced with CVaR) for Ising ferromagnetic 1D instances 
$$ H = \sum_{i=1}^{L-1} J_{i,i+1} \sigma_i \sigma_{i+1} + \sum_{i=1}^L h_i \sigma_i $$
where $L$ is the total number of spins $\mathbf{\sigma} = (\sigma_1, \cdots, \sigma_L)$, $J_{i,i+1}$ is the couplings between the $i$-th spin and the spin $i+1$, $h_i$ is the external field for the $i$-th spins.

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

<br>

Run with default parameters
```bash
python run.py --qubits 6 -vv
```

For the others parameters, take a look to the help
```bash
python run.py --help
```

<br>

## Data Availability

We release all the data considered in our study in a [Zenodo repository](https://doi.org/10.5281/zenodo.8223528). To run the notebook `paper-figure.ipynb` you first need to download the data cited above.
