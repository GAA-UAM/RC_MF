# RC_MF
RC-MF is a two-stage recommender framework that improves biased matrix factorization through regularized residual calibration.



# License
The package is licensed under the MIT license.

# Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/GAA-UAM/RC_MF.git
cd RC_MF

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Usage

## Basic command

Run experiments from the repository root:

```bash
python experiment_runner.py \
    --dataset DATASET_NAME \
    --epochs 50 \
    --seed 0 \
    --models rc_mf
```

Replace `DATASET_NAME` with the identifier of a dataset supported by the data loader.

## Run RC-MF without Bayesian optimization

When neither `--use_bo` nor `--use_gridsearch` is specified, the model is trained using the default parameters defined in:

```text
config/model_search_spaces.yml
```

Example:

```bash
python experiment_runner.py \
    --dataset Beauty \
    --epochs 100 \
    --seed 0 \
    --models rc_mf
```


## Run RC-MF with Bayesian optimization

Use the `--use_bo` option to tune the model hyperparameters through Bayesian optimization:

```bash
python experiment_runner.py \
    --dataset Beauty \
    --epochs 100 \
    --seed 0 \
    --use_bo \
    --models rc_mf
```

The search space and Bayesian-optimization budget are defined in:

```text
config/model_search_spaces.yml
```

# Citations
If you use RC-MF in your research or work, please consider citing this project using the following citation format.

```yml
@article{emami2026rcmf,
  title   = {Residual Correction Learning for Matrix Factorization},
  author  = {Emami, Seyedsaman and Bellogin, Alejandro and Hern{\'a}ndez-Lobato, Daniel},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## Authors

- Seyedsaman Emami
- Alejandro Bellogín
- Daniel Hernández-Lobato

Grupo de Aprendizaje Automático  
Universidad Autónoma de Madrid
