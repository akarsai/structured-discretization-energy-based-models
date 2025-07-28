# Structure-Preserving Discretization and Model Reduction for Energy-Based Models

This repository contains the code to the paper

R. Altmann, A. Karsai and P. Schulze, Structure-Preserving Discretization and Model Reduction for Energy-Based Models

## Reproducing our results
The first step is to install Python with version `>=3.12.5`.
We recommend using a virtual environment for this.
Using [pyenv](https://github.com/pyenv/pyenv) with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv), the steps are as follows:

<details>
<summary><b>How to install pyenv and pyenv-virtualenv</b></summary>
<br>

```bash
## install pyenv
# automatic installer
curl -fsSL https://pyenv.run | bash
# or macos or linux with homebrew:
#     brew install pyenv
# now make pyenv available in the shell (this assumes you use zsh. if you use another shell, please consult the pyenv manual)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# restart the shell
exec "$SHELL"

## install pyenv-virtualenv
# download and install
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
# or macos or linux with homebrew:
#     brew install pyenv-virtualenv
# add to path
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
# restart the shell
exec "$SHELL"
```
</details>

```bash
# this assumes pyenv is available in the environment
pyenv install --list | grep " 3\.[19]" # get all available versions, only starting with 3.1x or 3.9x
pyenv install 3.12.5 # choose 3.12.5 for example
pyenv virtualenv 3.12.5 sd-ebm # creates environment 'sd-ebm' with version 3.12.5
pyenv activate sd-ebm # activate virtual environment
```

The next step is to clone this repository, install the necessary requirements located in `requirements.txt`, and set the `PYTHONPATH` variable accordingly.
```bash
cd ~ # switch to home directory
git clone https://github.com/akarsai/structured-discretization-energy-based-models.git
cd structured-discretization-energy-based-models
pip install --upgrade pip # update pip
pip install -r requirements.txt # install requirements
export PYTHONPATH="${PYTHONPATH}:~/structured-discretization-energy-based-models" # add folder to pythonpath
```

Now, we can run the scripts in the [`plots`](./plots) folder to reproduce the figures in the paper.
The generated plots will be put in the directory [`results`](./results) as `.pgf` and `.png` files.
```bash
# the computations for the cahn-hilliard model can take a while
python plots/acdc_hamiltonian.py
python plots/cahn_hilliard_state.py
python plots/convergence.py
python plots/energybalance.py
```

If the code does not run, this could be due to recent changes in the dependencies. In this case, try changing `>=` in `requirements.txt` to `==`.



## Some hints
- Throughout the codebase, the time index is always at position `0`. The state `z` thus is stored in an array with shape `z.shape == (number_of_timepoints, dimension)`.
- Since the implementation uses the algorithmic differentiation capabilities of JAX, the implementations of all functions need to be written in a JAX-compatible fashion. The provided examples should be a good starting point.
- In case of questions, feel free to reach out.