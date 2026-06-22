# Code and Data of "How Far Can Machine Learning Guide Catalyst Discovery?"

This repository contains the code and datasets required to reproduce the experiments described in the paper. In addition to rerunning the main training workflows, the repository already includes archived experimental results under `experiments/` and the notebook used to generate the main-text figures at `notebooks/paper_plots.ipynb`.

## Repository layout

- `Dataset/`: source datasets used in the study, including the virtual catalyst datasets.
- `src/`: training and analysis scripts.
- `experiments/`: saved outputs from completed experiments.
- `notebooks/paper_plots.ipynb`: notebook for generating the plots shown in the main text.

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
  --augmentations yes \
```

Notes:

- the example reproduces the setup from the paper with 50 randomly sampled hyper-parameter combinations for 5-fold cross validation (CV), using 49 catalysts for training and 10 for testing, split by catalyst composition and repeated for the five given seeds
- `--data_path` the path to the dataset csv file (for example `./Dataset/OCM-NguyenEtAl_with_descriptors.csv` when running from the repository root). It is required.
- `--seeds` accepts either individual integers or ranges such as `1:10`. Default: 1
- `--split_strategy` can be either `catalyst` (for catalyst composition-based splits) or `catalyst_random` (for splitting data of `--n_train_catalysts` catalysts into 80% training and 20% test sets randomly, ignoring `--n_test_catalysts`). Default: catalyst
- `--n_folds` set the number of folds for CV (must be >1). Default: 5
- `--cross_val_params` the amount of random hyper-parameter combinations sampled for CV (must be >0). Default: 50
- `--feature_sets` can be any combination of `base+atom_numbers+support` (process conditions and categorical features for catalyst composition and support), `base+descriptors` (the process conditions and our continuous physicochemical descriptors for catalyst composition and support), and `all` (all of the aforementioned features). Default: base+atom_numbers+support base+descriptors all
- `--augmentations` can be any combination of `yes` and `no`. Default: yes no
- if multiple values are provided for `--feature_sets` and `--augmentations`, the script will run experiments for all possible combinations
- `--conditions` can be `None`, `temp_single`, or `ch4_o2_ratio`. If not `None`, the script will split the data based on target process condition values of temperature or ch4/o2 ratio instead of catalyst composition (note that `--split_strategy` must still be set to `catalyst` in that case). Default: None
- Results are written under the repository root to a folder named from the selected CV setting, in this example `xgb_cv_random_50/`. It contains subfolders named by the training data volume (and the target process condition value if `--conditions` is not `None`). A single csv file containing all results can be compiled by calling `python src/gather_results.py ./xgb_cv_random_50`.

### 2. Virtual catalyst OCM XGBoost experiments

The main workflow for the virtual catalyst datasets is `src/virtual_xgb.py`.

Example:

```bash
python src/virtual_xgb.py \
  --data_path Dataset/virtual_catalyst_datasets/Top59.csv \
  --n_train_catalysts 49 \
  --seeds 1 2 3 4 5
```

Notes:

- `--data_path` and `--n_train_catalysts` are required.
- The path to the `Full` dataset as well as the filtered versions `AnyActive`, `MostlyActive`, and `Top59` can be provided via `--data_path` (see `Dataset/virtual_catalyst_datasets`).
- The script uses catalyst-based splitting with 5-fold cross-validation with 50 randomly sampled hyper-parameter combinations internally and a fixed test size of 10 catalysts.
- Outputs are written under the repository root to a run folder such as `virtual_xgb_random_50/<n_train_catalysts>/`.

## Reproducing paper plots

The notebook at `notebooks/paper_plots.ipynb` reads the provided results in `experiments/` and generates the plots used in the paper. After setting up the environment, launch Jupyter and open the notebook:

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

For completeness, we also provide training scripts used in our model selection, where we compare the selected XGBoost to Random Forest and a simple Neural Network.
The script for Random Forest (`src/ocm_rf.py`) is invoked just as the XGBoost example described above, accepting the same arguments.
Since the training routine for Neural Network is slightly different (see Supplement of the paper), some arguments differ. The call is:

```bash
python src/ocm_nn.py \
  --data_path Dataset/OCM-NguyenEtAl_with_descriptors.csv \
  --seed 1 \
  --n_train_catalysts 39 \
  --n_val_catalysts 10 \
  --n_test_catalysts 10 \
  --val_params 25 \
  --feature_sets base+descriptors \
  --augmentations yes \
```

It only takes a single `--seed` (default: 1) and requires the number of validation catalysts `--n_val_catalysts` used during training (for adapting the learning rate, selecting the best model etc.). The script uses catalyst-based splitting and trains `--val_params` models, selecting the one with lowest validation loss for evaluation on the test set.

## Citation

If you use this code or data from this repository, please cite the corresponding publication:

```
Citation will be added upon publication.
```

If you use the enriched experimental dataset (`Datasets/OCM-NguyenEtAl_with_descriptors.csv`) please also cite its original source:

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