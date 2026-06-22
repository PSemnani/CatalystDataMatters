import random
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from utils import (
    BASE_PROCESS,
    ATOM_NUMBERS,
    SUPPORT,
    DESCRIPTORS,
    scale_data,
    split_data,
    augment_data,
    get_cross_validation_param_sets,
)


# Simple Dataset wrapper
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Simple MLP for regression
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_nn(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    epochs=20000,
    batch_size=1024,
    lr=1e-3,
    min_lr=1e-6,
    weight_decay=1e-5,
    hidden_dims=(256, 128, 64),
    dropout=0.05,
    device=None,
    patience=100,
    factor=0.8,
    ema_decay=0.999,
    ema_warmup=1,
    early_stopping=2000,
    augment_data_flag=False,
):
    assert ema_warmup >= 1, "ema_warmup must be at least 1"

    # Data augmentation (permutations of M1, M2, M3 related features)
    if augment_data_flag:
        X_train, y_train, X_val, y_val, X_test, y_test, _ = augment_data(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_cols,
        )

    # Scaling
    X_train, X_val, X_test, scaler = scale_data(
        X_train,
        X_val,
        X_test,
        feature_cols,
        passthrough_cols=[],
    )

    # Datasets & loaders
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    model = SimpleMLP(
        input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "lr": []}
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # EMA and validation
        with torch.no_grad():
            if epoch <= ema_warmup:
                ema_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                for k, v in model.state_dict().items():
                    ema_state[k] = (
                        ema_decay * ema_state[k] + (1.0 - ema_decay) * v.detach().cpu()
                    )
                orig_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                ema_to_device = {k: ema_state[k].to(device) for k in ema_state}
                model.load_state_dict(ema_to_device)
            model.eval()

            val_losses = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())

            avg_val = (
                float(np.mean(val_losses)) if len(val_losses) > 0 else float("inf")
            )
            # improvement check and best_state (store EMA weights)
            if avg_val < best_val_loss - 1e-5:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in ema_state.items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # restore original weights for continued training
            if epoch > ema_warmup:
                orig_to_device = {k: orig_state[k].to(device) for k in orig_state}
                model.load_state_dict(orig_to_device)

        avg_train = (
            float(np.mean(train_losses)) if len(train_losses) > 0 else float("nan")
        )
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # scheduler step
        scheduler.step(avg_val)

        if epoch % patience == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        if optimizer.param_groups[0]["lr"] < min_lr:
            print(f"Stopping early at epoch {epoch} due to reaching min_lr")
            break

        if epochs_no_improve >= early_stopping:
            print(
                f"Early stopping after {epoch} epochs (no validation improvement for {epochs_no_improve} epochs)."
            )
            break

    # load best model (EMA weights) if available
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # evaluate on test set
    preds_all = []
    ys_all = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().ravel()
            preds_all.append(out)
            ys_all.append(yb.numpy().ravel())
    preds_all = np.concatenate(preds_all)
    ys_all = np.concatenate(ys_all)

    test_mse = float(np.mean((preds_all - ys_all) ** 2))
    test_mae = float(np.mean(np.abs(preds_all - ys_all)))
    test_rmse = float(np.sqrt(test_mse))
    ss_res = np.sum((preds_all - ys_all) ** 2)
    ss_tot = np.sum((ys_all - np.mean(ys_all)) ** 2)
    test_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    if augment_data_flag:
        # take mean of premutation predictions for final test metrics
        _preds_all = preds_all.reshape(-1, 6)
        _preds_all = np.mean(_preds_all, axis=1)
        _ys_all = ys_all[::6]
        test_mse = float(np.mean((_preds_all - _ys_all) ** 2))
        test_mae = float(np.mean(np.abs(_preds_all - _ys_all)))
        test_rmse = float(np.sqrt(test_mse))
        ss_res = np.sum((_preds_all - _ys_all) ** 2)
        ss_tot = np.sum((_ys_all - np.mean(_ys_all)) ** 2)
        test_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        print(f"Results after averaging over permutations to get predictions:")
        print(f"Test R2: {test_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    results = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "history": history,
        "test_metrics": {
            "mse": test_mse,
            "mae": test_mae,
            "rmse": test_rmse,
            "r2": test_r2,
        },
        "preds_test": preds_all,
        "y_test": ys_all,
    }
    return results


def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # use log scaling on y-axis for loss plot
    plt.yscale("log")
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    plt.tight_layout()
    return plt.gcf()


def plot_test_results(y_true, y_pred, accumulate_permutations=False):
    if accumulate_permutations:
        # average predictions over permutations
        y_pred = y_pred.reshape(-1, 6).mean(axis=1)
        y_true = y_true[::6]
    # compute R2, MAE, RMSE
    r2 = 1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    max_val = 22.5  # max(max(y_true), max(y_pred))
    min_val = -2.5  # min(min(y_true), min(y_pred))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.plot([min_val, max_val], [min_val, max_val], "r--")  # y=x line
    plt.xlabel("True C2y")
    plt.ylabel("Predicted C2y")
    plt.title(f"Test set predictions (R²={r2:.3f}, MAE={mae:.3f}, MSE={mse:.3f})")
    plt.grid(True)
    return plt.gcf(), r2, mae, mse


def main(
    data_path,
    seed,
    n_train_catalysts,
    n_val_catalysts,
    n_test_catalysts,
    feature_sets,
    augmentations,
    val_params,
):
    # read data
    df = pd.read_csv(data_path)
    # storage path for results
    results_path = Path(f"./NN/{n_train_catalysts+n_val_catalysts}/")
    results_path.mkdir(parents=True, exist_ok=True)
    # create empty dataframe to store results
    results_rows = []

    for feature_set in feature_sets:
        # run experiments per feature set
        if feature_set == "base+descriptors":
            feature_cols = BASE_PROCESS + DESCRIPTORS
            file_identifier = "descriptors"
        elif feature_set == "base+atom_numbers+support":
            feature_cols = BASE_PROCESS + ATOM_NUMBERS + SUPPORT
            file_identifier = ""
        else:  # all features
            feature_cols = BASE_PROCESS + ATOM_NUMBERS + DESCRIPTORS + SUPPORT
            file_identifier = "all"
        target_col = "C2y"
        split_strategy = "catalyst"

        # initialize random seed (for data splitting, we want the same splits each run)
        rng = random.Random(seed)

        # Data splitting
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            df,
            feature_cols,
            target_col,
            split_strategy,
            rng,
            n_train_catalysts=n_train_catalysts,
            n_val_catalysts=n_val_catalysts,
            n_test_catalysts=n_test_catalysts,
            return_indices=False,
        )

        for augm in augmentations:
            # Train neural network
            print(f"Training Neural Network model for feature set: {feature_set}...")
            print(f"Training {'with' if augm else 'without'} data augmentation...")
            choices = get_cross_validation_param_sets(f"nn_{val_params}", seed)
            val_losses = []
            test_mses = []
            test_r2s = []
            test_maes = []
            timings = []
            for n, nn_args in enumerate(choices):
                print(f"NN training run {n+1}/{len(choices)} with params: {nn_args}")
                start_time = time()
                nn_results = train_nn(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    feature_cols,
                    device="cuda",
                    augment_data_flag=augm,
                    **nn_args,
                )
                _, r2, mae, mse = plot_test_results(
                    nn_results["y_test"],
                    nn_results["preds_test"],
                    accumulate_permutations=augm,
                )
                elapsed_time = time() - start_time
                print(f"NN training run {n+1} completed in {elapsed_time:.2f} seconds.")
                timings.append(elapsed_time)
                val_losses.append(np.min(nn_results["history"]["val_loss"]))
                test_mses.append(mse)
                test_r2s.append(r2)
                test_maes.append(mae)
            best_model = np.argmin(val_losses)
            r2 = test_r2s[best_model]
            mae = test_maes[best_model]
            mse = test_mses[best_model]
            timing = timings[best_model]
            results_rows.append(
                {
                    "model_type": "neural_network",
                    "feature_set": feature_set,
                    "augmentation": "yes" if augm else "no",
                    "n_train_catalysts": n_train_catalysts,
                    "n_val_catalysts": n_val_catalysts,
                    "n_test_catalysts": n_test_catalysts,
                    "seed": seed,
                    "r2": r2,
                    "mae": mae,
                    "mse": mse,
                    "training_time": timing,
                }
            )
            if len(choices) > 1:
                results_rows[-1].update({
                    **{f"val_loss_{i}": val_losses[i] for i in range(len(choices))},
                    **{f"mae_{i}": test_maes[i] for i in range(len(choices))},
                    **{f"mse_{i}": test_mses[i] for i in range(len(choices))},
                    **{f"r2_{i}": test_r2s[i] for i in range(len(choices))},
                    **{f"timing_{i}": timings[i] for i in range(len(choices))},
                })

    # save results dataframe to csv
    results_df = pd.DataFrame(results_rows)
    summary_path = results_path / "results_summary.csv"
    if summary_path.exists():
        results_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training/evaluation")
    # add argument for datapath
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input data file",
    )
    # add argument for seed
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for running the experiment (default 1)",
    )
    # add arguments for n_train_catalysts, n_val_catalysts, n_test_catalysts
    parser.add_argument(
        "--n_train_catalysts",
        type=int,
        default=39,
        help="Number of training catalysts (default: 39)",
    )
    parser.add_argument(
        "--n_val_catalysts",
        type=int,
        default=10,
        help="Number of validation catalysts (default: 10)",
    )
    parser.add_argument(
        "--n_test_catalysts",
        type=int,
        default=10,
        help="Number of test catalysts (default: 10)",
    )
    parser.add_argument(
        "--feature_sets",
        type=str,
        nargs="+",
        default=["base+atom_numbers+support", "base+descriptors", "all"],
        help="Feature sets to run (default: all three sets)",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        default=["yes", "no"],
        help="Augmentation options to run (default: both yes and no)",
    )
    parser.add_argument(
        "--val_params",
        type=int,
        default=25,
        help="Number of hyper-parameter combinations randomly sampled for validation (default: 25)",
    )

    args = parser.parse_args()
    assert args.val_params > 0, "val_params must be a positive integer"
    augmentations = [augm.lower() in ["yes", "1"] for augm in args.augmentations]
    main(
        args.data_path,
        args.seed,
        args.n_train_catalysts,
        args.n_val_catalysts,
        args.n_test_catalysts,
        args.feature_sets,
        augmentations,
        args.val_params,
    )
