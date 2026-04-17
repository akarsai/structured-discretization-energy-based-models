# Code for the paper *"Structure-Preserving Discretization and Model Reduction for Energy-Based Models"*

This repository contains the code to the paper

R. Altmann, A. Karsai and P. Schulze, Structure-Preserving Discretization and Model Reduction for Energy-Based Models

## Reproducing the results

1. Install [`uv`](https://github.com/astral-sh/uv) by following the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository and switch folder: 
```shell
git clone https://github.com/akarsai/structured-discretization-energy-based-models.git && cd structured-discretization-energy-based-models
``` 
3. **Optional:** Download [a compatible dataset](https://doi.org/10.5281/zenodo.19358955) to use precalculated results, place downloaded files in `results/pickle` directory.
4. Run experiments (the plots will be placed in the `results/figures` directory):
```shell
uv run python plots/cahn_hilliard_state.py
```
```shell
uv run python plots/convergence.py
```
```shell
uv run python plots/energybalance.py
```



## Some hints
- Throughout the codebase, the time index is always at position `0`. The state `z` thus is stored in an array with shape `z.shape == (number_of_timepoints, dimension)`.
- Since the implementation uses the algorithmic differentiation capabilities of JAX, the implementations of all functions need to be written in a JAX-compatible fashion. The provided examples should be a good starting point.
- In case of questions, feel free to reach out.