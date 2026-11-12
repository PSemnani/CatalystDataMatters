# Code and Data of "How Far Can Machine Learning Guide Catalyst Discovery?"

This repository contains the code and datasets required to reproduce the experiments described in the paper. The experimental OCM dataset by Nguyen et al. is not included and has to be downloaded separately (see [Preparing the experimental OCM dataset](#preparing-the-experimental-ocm-dataset)). In addition to rerunning the training workflows, the repository already includes the results of all experiments under `experiments/` and the notebook used to generate the figures of the main text and the Supplement at `notebooks/paper_plots.ipynb`.

## Repository layout

- `Dataset/`: datasets used in the study, i.e. the virtual catalyst datasets (`Dataset/virtual_catalyst_datasets/`), the precomputed XenonPy descriptors of the experimental catalysts (`Dataset/xenonpy_descriptors.csv`), and the location for the experimental OCM dataset (to be downloaded separately).
- `src/`: training and analysis scripts.
- `src/data_scripts/`: scripts for preparing the experimental OCM dataset.
- `experiments/`: results of the experiments in the paper and the generated figures (`experiments/figures/`).
- `notebooks/paper_plots.ipynb`: notebook for generating the figures and tables of the paper from the results in `experiments/`.

## Environment setup

The environment used for the experiments is captured in `requirements.txt`.

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run all commands from the repository root.

## Preparing the experimental OCM dataset

The experimental OCM dataset by Nguyen et al. is not included in this repository, since it may not be redistributed. To obtain it and prepare it for our experiments:

1. Go to https://cads.eng.hokudai.ac.jp/datamanagement/ and select "High-throughput OCM data".
2. Click "Download" on the top right of the table. Please note the terms of use stated on the website.
3. Store the downloaded csv file as `Dataset/OCM-NguyenEtAl.csv` inside the repository.
4. Add the physicochemical descriptors of the catalyst composition and the support surface area used in our study:

```bash
python src/data_scripts/build_descriptor_data.py
```

This creates `Dataset/OCM-NguyenEtAl_with_descriptors.csv`, which is the dataset used in all experiments with the experimental OCM data below. The script also sets the molar fractions (`M1_mol%`, `M2_mol%`, `M3_mol%`) of catalysts with inconsistent compositions across rows to the composition consistent with their metal loadings (see the docstring of the script for details). You can check that you obtained the same dataset as used in the paper with `sha256sum Dataset/OCM-NguyenEtAl_with_descriptors.csv`, which should print `358ed33eb3f7a51548bed16ec3a22d7825fe5f82b8de669e84c660f73bf35756`.

## Running the main scripts

### 1. OCM XGBoost experiments

The main workflow for the experimental OCM dataset is `src/ocm_xgb.py`.

Example:

```bash
python src/ocm_xgb.py \
  --data_path Dataset/OCM-NguyenEtAl_with_descriptors.csv \
  --seeds 1 2 3 4 5 \
  --n_train_catalysts 49 \
  --n_test_catalysts 10 \
  --split_strategy catalyst \
  --n_folds 5 \
  --cross_val_params 50 \
  --conditions None \
  --feature_sets base+descriptors \
  --augmentations yes
```

Notes:

- the example reproduces the setup from the paper with 50 randomly sampled hyper-parameter combinations for 5-fold cross validation (CV), using 49 catalysts for training and 10 for testing, split by catalyst composition and repeated for the five given seeds
- `--data_path` the path to the dataset csv file (for example `Dataset/OCM-NguyenEtAl_with_descriptors.csv`). It is required.
- `--seeds` accepts either individual integers or ranges such as `1:10` (inclusive). Default: 1
- `--n_train_catalysts` and `--n_test_catalysts` set the number of training and test catalysts. Default: 49 and 10
- `--split_strategy` can be either `catalyst` (for catalyst composition-based splits) or `catalyst_random` (for splitting the data of `--n_train_catalysts` randomly selected catalysts into 80% training and 20% test data points, ignoring `--n_test_catalysts`). Default: catalyst
- `--n_folds` sets the number of folds for CV (must be >1). Default: 5
- `--cross_val_params` the number of random hyper-parameter combinations sampled for CV (must be >0). Default: 50
- `--feature_sets` can be any combination of `base+atom_numbers+support` (process conditions, atomic numbers of the metals, molar fractions, and support ID), `base+descriptors` (process conditions and our physicochemical descriptors of the metals and the support), `one_hot` (one-hot encoded elements per metal position and supports), `invariant` (molar fractions per element and one-hot encoded supports, independent of the order of the metals), and `xenonpy` (XenonPy compositional descriptors of the metals and the support from `Dataset/xenonpy_descriptors.csv`). Augmentation is skipped for `invariant` and `xenonpy`. Default: base+descriptors
- `--augmentations` can be any combination of `yes` and `no` (augmentation of the training data with all distinct permutations of the metal positions). Default: yes no
- if multiple values are provided for `--feature_sets` and `--augmentations`, the script will run experiments for all possible combinations
- `--conditions` can be `None`, `temp_single`, or `ch4_o2_ratio`. If not `None`, the script holds out each target value of the temperature (`temp_single`) or CH4/O2 ratio (`ch4_o2_ratio`) in turn instead of splitting by catalyst composition only: the test set contains the data points of `--n_test_catalysts` catalysts at the target value, and the training set contains all data points at other values plus the data points of `--n_train_catalysts` catalysts at the target value (so `--n_train_catalysts 0` is possible). `--split_strategy` must be `catalyst` in that case. Default: None
- `--compute_shap` computes the SHAP values of the trained models and `--compute_permutation_importance` the permutation importance of single features and of the feature groups process conditions and catalyst composition on the test set (with `--permutation_repeats` permutations per feature or group, default: 10). Both are off by default.
- `--store_models` and `--store_plots` additionally store the trained models and plots of the test predictions. Both are off by default.
- Results are written to a folder named from the selected CV setting in the current working directory, in this example `xgb_cv_random_50/`. It contains subfolders named by the number of training catalysts with the results (`training_results.csv`) and the data splits. A single csv file containing all results can be compiled by calling `python src/gather_results.py xgb_cv_random_50`, which writes `xgb_cv_random_50/gathered_results.csv` (duplicate runs, e.g. from repeated seeds, are removed unless `--no_dedup` is given; an existing `gathered_results.csv` is not overwritten).

The results in `experiments/` were obtained with seeds `201:300` (`201:220` for `catalyst_random`) and the following settings (all other arguments as in the example above):

| Results in `experiments/` | Data | Arguments |
| --- | --- | --- |
| `split_catalyst/` | `OCM-NguyenEtAl_with_descriptors.csv` | `--n_train_catalysts` 1 to 49, `--feature_sets base+descriptors --augmentations yes no --compute_shap --compute_permutation_importance`; for 49 training catalysts also all other feature sets and for 10 training catalysts also `--feature_sets base+atom_numbers+support --augmentations no` |
| `split_random/` | `OCM-NguyenEtAl_with_descriptors.csv` | as `split_catalyst/` with `--split_strategy catalyst_random` |
| `temperature/` | `OCM-NguyenEtAl_with_descriptors.csv` | `--conditions temp_single`, `--n_train_catalysts` 0 to 49, `--feature_sets base+descriptors --augmentations yes`; for 49 training catalysts also `--feature_sets base+atom_numbers+support --augmentations no` |
| `ratio_ch4o2/` | `OCM-NguyenEtAl_with_descriptors.csv` | as `temperature/` with `--conditions ch4_o2_ratio` |
| `fixed_condition/` | `OCM-NguyenEtAl_with_descriptors_single_condition.csv` (created with `python src/data_scripts/filter_single_condition.py`) | `--n_train_catalysts 49`, all feature sets, `--augmentations yes no` |

### 2. Virtual catalyst OCM XGBoost experiments

The main workflow for the virtual catalyst datasets is `src/virtual_xgb.py`.

Example:

```bash
python src/virtual_xgb.py \
  --data_path Dataset/virtual_catalyst_datasets/top59.csv \
  --n_train_catalysts 49 \
  --seeds 1 2 3 4 5
```

Notes:

- `--data_path` and `--n_train_catalysts` are required. `--seeds` takes a list of integers. Default: 1 2 3 4 5
- The full virtual catalyst dataset (`full.csv`) as well as the filtered versions `top59.csv` and `top160.csv` (the 59 resp. 160 catalysts with the highest mean yield) can be provided via `--data_path` (see `Dataset/virtual_catalyst_datasets`).
- The script uses catalyst-based splitting with 5-fold cross-validation with 50 randomly sampled hyper-parameter combinations internally and a fixed test size of 10 catalysts.
- `--train_pool_path` and `--test_pool_path` optionally restrict the catalysts drawn for training and testing to those found in the catalyst ID column of another csv file (must be a subset of the catalysts in `--data_path`). If omitted, all catalysts in `--data_path` are used. For example, `--data_path Dataset/virtual_catalyst_datasets/full.csv --test_pool_path Dataset/virtual_catalyst_datasets/top59.csv` trains on the full dataset but tests only on catalysts with high mean yield.
- `--compute_shap` additionally computes SHAP values and `--plot_performance` stores plots of the predictions. Both are off by default.
- Outputs are written to a run folder in the current working directory such as `virtual_xgb_random_50/<n_train_catalysts>/`.

## Reproducing paper plots

The notebook at `notebooks/paper_plots.ipynb` reads the provided results in `experiments/` and generates the figures (stored in `experiments/figures/`) and tables of the paper. After setting up the environment and preparing the experimental OCM dataset (some figures use the data directly), launch Jupyter and open the notebook:

```bash
jupyter lab
```

## Using our splitting strategies with custom data

We provide utility functions in `src/utils.py` which can also be used to split custom catalyst data according to our proposed catalyst composition-based and process condition-based splitting strategies.
The data has to be available as a pandas dataframe (e.g. by loading a csv) that has a column which identifies catalysts by composition (e.g. a string describing the composition or an index associated with a specific composition).
We assume that the dataframe is stored in the variable `df` and that the name of the column is stored in the variable `cat_id`.
Furthermore, you should have a list of the names of the columns used as input features stored in the variable `feature_columns` and the name of the target column stored in `target_column`.
Given these requirements, the catalyst composition-based splitting can be achieved with:

```python
import random
from src.utils import split_data

# initialize random state
seed = 1 # set a seed here
rng = random.Random(seed)

# split data in pandas data frame df
X_train, y_train, _, _, X_test, y_test = split_data(
    df,
    feature_columns,
    target_column,
    split_strategy="catalyst",
    rng=rng,
    n_train_catalysts=49,
    n_test_catalysts=10,
    conditions={},
    catalyst_name_column=cat_id,
)
```

Process condition-based splitting can be done by specifying a nested dictionary with the name of the target process condition and the desired value as `conditions`, for example for a temperature of 700°C with our OCM dataset, where the column for temperature is called `"Temp"`, we would use:

```python
conditions={"Temp": {"==": 700}}
```

Note that if conditions are provided, all data points not at the target condition value (e.g. at all temperatures not 700°C) are in the train set.
Additionally, the data points at the target condition value (e.g. at 700°C) for `n_train_catalysts` catalysts are added to the training set.
The test set consists only of data points at the target condition value (e.g. at 700°C) for `n_test_catalysts` catalysts.
For more details on the splitting strategies, please check the paper.

## Scripts for Random Forest and Neural Network

For completeness, we also provide the training scripts used in our model selection, where we compare XGBoost to Random Forest and a simple Neural Network.
The script for Random Forest (`src/ocm_rf.py`) is invoked just as the XGBoost example described above and accepts the same arguments, except that only the feature sets `base+atom_numbers+support` and `base+descriptors` are available and SHAP values, permutation importance, models and plots are not computed or stored. Results are written to `rf_cv_random_<cross_val_params>/<n_train_catalysts>/`.
Since the training routine for the Neural Network is slightly different (see Supplement of the paper), some arguments differ. The call is:

```bash
python src/ocm_nn.py \
  --data_path Dataset/OCM-NguyenEtAl_with_descriptors.csv \
  --seed 1 \
  --n_train_catalysts 39 \
  --n_val_catalysts 10 \
  --n_test_catalysts 10 \
  --val_params 25 \
  --feature_sets base+descriptors \
  --augmentations yes
```

It only takes a single `--seed` (default: 1) and requires the number of validation catalysts `--n_val_catalysts` (default: 10) used during training (for adapting the learning rate, selecting the best model etc.). The script uses catalyst-based splitting and trains `--val_params` models (default: 25), selecting the one with the lowest validation loss for evaluation on the test set. The feature sets `base+atom_numbers+support` and `base+descriptors` are available. Results are written to `NN/<n_train_catalysts + n_val_catalysts>/`.

The results in `experiments/model_selection/` were obtained with seeds 201 to 300, `--feature_sets base+atom_numbers+support --augmentations no`, and 10 or 49 training catalysts (`--n_train_catalysts 5 --n_val_catalysts 5` or `--n_train_catalysts 39 --n_val_catalysts 10` for the Neural Network).

## Citation

If you use the code or data from this repository, please cite the corresponding publication:

> Semnani, P., Gebauer, N. W. A., Alves Sequeira, B., Zareghorbaei, S., Thybaut, J. W., Vermeire, F. H., & Mendes, P. S. F. (2026). *How Far Can Machine Learning Guide Catalyst Discovery?* ChemRxiv. https://doi.org/10.26434/chemrxiv.15005240

```bibtex
@article{semnani2026far,
  title   = {How Far Can Machine Learning Guide Catalyst Discovery?},
  author  = {Semnani, Parastoo and Gebauer, Niklas W. A. and Alves Sequeira, B{\'a}rbara and Zareghorbaei, Soroush and Thybaut, Joris W. and Vermeire, Florence H. and Mendes, Pedro S. F.},
  year    = {2026},
  howpublished = {ChemRxiv preprint},
  doi     = {10.26434/chemrxiv.15005240},
  url     = {https://chemrxiv.org/doi/full/10.26434/chemrxiv.15005240/v1}
}
```

If you use the experimental OCM dataset (`Dataset/OCM-NguyenEtAl.csv`) or our version enriched with descriptors (`Dataset/OCM-NguyenEtAl_with_descriptors.csv`), please also cite its original source:

```
@article{nguyen2019high,
  title={High-throughput experimentation and catalyst informatics for oxidative coupling of methane},
  author={Nguyen, Thanh Nhat and Nhat, Thuy Tran Phuong and Takimoto, Ken and Thakur, Ashutosh and Nishimura, Shun and Ohyama, Junya and Miyazato, Itsuki and Takahashi, Lauren and Fujima, Jun and Takahashi, Keisuke and others},
  journal={ACS Catalysis},
  volume={10},
  number={2},
  pages={921--932},
  year={2019},
  publisher={ACS Publications},
  DOI= {10.1021/acscatal.9b04293}
}
```


## License

This repository is distributed under the terms of the license in `LICENSE`.
