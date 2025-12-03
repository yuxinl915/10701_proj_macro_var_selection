import os
import gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


# ---------------------------
# Data loading
# ---------------------------


def load_data(path: str):
    df = pd.read_csv(path)
    print("Raw data shape:", df.shape)

    # ensure sorted by time
    if "eom" in df.columns:
        df = df.sort_values("eom").reset_index(drop=True)

    print("Shape after dropping outlier and sorting:", df.shape)

    # target
    y = df["ret_exc_lead1m"].to_numpy(dtype="float32")


    features_df = df.iloc[:, 2:]
    print("Feature columns:", features_df.columns.tolist())
    print("Feature matrix (before scaling) shape:", features_df.shape)
    X_raw = features_df.to_numpy(dtype="float32")

    # standardize features to mean 0, variance 1
    means = X_raw.mean(axis=0)
    vars_ = X_raw.var(axis=0)
    stds  = np.sqrt(vars_)

    # avoid division by zero for any constant feature
    stds_safe = stds.copy()
    stds_safe[stds_safe == 0] = 1.0
    X_scaled = (X_raw - means) / stds_safe

    del df, features_df, X_raw
    gc.collect()

    return X_scaled.astype("float32"), y.astype("float32")

# specific for train/test split
def load_train_test(path: str,
                    train_start="1990-01-01",
                    train_end="2020-12-31"):
    
    df = pd.read_csv(path)
    print("Raw data shape:", df.shape)


    # parse and sort by eom
    df["eom"] = pd.to_datetime(df["eom"])
    df = df.sort_values("eom").reset_index(drop=True)

    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    mask_train = (df["eom"] >= train_start) & (df["eom"] <= train_end)
    mask_test = df["eom"] > train_end


    # Target
    y = df["ret_exc_lead1m"].to_numpy(dtype="float32")

    # features: all columns after eom and label
    features_df = df.iloc[:, 2:]
    X_raw = features_df.to_numpy(dtype="float32")


    X_train_raw = X_raw[mask_train.values]
    y_train = y[mask_train.values]
    X_test_raw = X_raw[mask_test.values]
    y_test = y[mask_test.values]
    # for debug
    print("X_train_raw shape:", X_train_raw.shape)
    print("X_test_raw shape :", X_test_raw.shape)

    means = X_train_raw.mean(axis=0)
    vars_ = X_train_raw.var(axis=0) 
    stds = np.sqrt(vars_)
    stds_safe = stds.copy()
    stds_safe[stds_safe == 0] = 1.0

    X_train = (X_train_raw - means) / stds_safe
    X_test = (X_test_raw - means) / stds_safe

    del df, features_df, X_raw, X_train_raw, X_test_raw
    gc.collect()

    return (X_train.astype("float32"), y_train.astype("float32"),
            X_test.astype("float32"), y_test.astype("float32"))

# ---------------------------
# Model definition
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FFNRegressor(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Training helper
# ---------------------------

def train_one_model(
    X_train,
    y_train,
    X_val,
    y_val,
    lr: float,
    num_layers: int = 3,
    num_epochs: int = 10,
    batch_size: int = 256,
    hidden_dim: int = 64,
    verbose: bool = False,
):

    input_dim = X_train.shape[1]
    model = FFNRegressor(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()

    if verbose:
        print(f"[train_one_model] lr={lr:.1e}, layers={num_layers}, "
              f"epochs={num_epochs}, val MSE={val_loss:.6f}")

    return val_loss, model


# ---------------------------
# Cross-validation for learning rate
# ---------------------------

def cross_validate_learning_rate(
    X,
    y,
    num_layers: int = 3,
    learning_rates=None,
    n_splits: int = 3,
    num_epochs: int = 10,
    batch_size: int = 256,
    hidden_dim: int = 64,
):
    if learning_rates is None:
        learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    lr_val_losses = {}

    for lr in learning_rates:
        fold_losses = []
        print(f"\n=== CV for lr={lr:.1e} (layers={num_layers}, epochs={num_epochs}) ===")
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            val_loss, _ = train_one_model(
                X_train,
                y_train,
                X_val,
                y_val,
                lr=lr,
                num_layers=num_layers,
                num_epochs=num_epochs,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                verbose=False,
            )
            fold_losses.append(val_loss)
            print(f"  Fold {fold + 1}, val MSE = {val_loss:.6f}")

        mean_loss = float(np.mean(fold_losses))
        lr_val_losses[lr] = mean_loss
        print(f"Mean val MSE for lr={lr:.1e}: {mean_loss:.6f}")

    best_lr = min(lr_val_losses, key=lr_val_losses.get)
    print("\nBest learning rate from CV:", best_lr)
    print("All CV results:", lr_val_losses)

    return best_lr, lr_val_losses


def plot_lr_curve(lr_val_losses, title="Learning rate vs validation MSE", save_path=None):

    lrs = sorted(lr_val_losses.keys())
    mses = [lr_val_losses[lr] for lr in lrs]

    plt.figure(figsize=(6, 4))
    plt.plot(lrs, mses, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Mean validation MSE")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved LR vs MSE plot to: {save_path}")
    else:
        plt.show()


# ---------------------------
# Single train function
# ---------------------------

def run_single_train(
    X,
    y,
    num_layers: int = 3,
    num_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    hidden_dim: int = 64,
    summary_path: str = "train_summary.csv",
):
    
    print(f"\nRunning single train: num_layers={num_layers}, num_epochs={num_epochs}, lr={lr}")
    final_loss, model = train_one_model(
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        lr=lr,
        num_layers=num_layers,
        num_epochs=num_epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        verbose=True,
    )
    print("Final training MSE on full data:", final_loss)

    summary = {
        "num_layers": num_layers,
        "num_epochs": num_epochs,
        "lr": lr,
        "final_train_mse": final_loss,
    }

    df_summary = pd.DataFrame([summary])
    write_header = not os.path.exists(summary_path)
    df_summary.to_csv(summary_path, mode="a", index=False, header=write_header)
    print(f"Appended training summary to: {summary_path}")

    return summary, model



if __name__ == "__main__":
    # print("Using device:", device)

    # # 1. load data
    # X, y = load_data("data_new.csv.gz")

    # # 2. choose number of layers and epochs
    # num_layers = 3 
    # num_epochs_cv = 10     
    # num_epochs_final = 10   

    # # 3. cross-validate learning rate
    # best_lr, lr_cv_results = cross_validate_learning_rate(
    #     X,
    #     y,
    #     num_layers=num_layers,
    #     learning_rates=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    #     n_splits=3,
    #     num_epochs=num_epochs_cv,
    #     batch_size=256,
    #     hidden_dim=64,
    # )

    # # 4. Plot LR vs mean validation MSE
    # plot_lr_curve(
    #     lr_cv_results,
    #     title=f"LR vs val MSE (layers={num_layers}, epochs={num_epochs_cv})",
    #     save_path="lr_vs_val_mse.png",  # or None to just show
    # )

    # 5. sinlge train on full data with best learning rate
    # train_summary, final_model = run_single_train(
    #     X,
    #     y,
    #     num_layers=num_layers,
    #     num_epochs=num_epochs_final,
    #     lr=best_lr,
    #     batch_size=256,
    #     hidden_dim=64,
    #     summary_path="train_summary.csv",
    # )

    # print("\nTraining summary:", train_summary)

    # print("Using device:", device)

    # 1. load data
    X_train, y_train, X_test, y_test = load_train_test("data_new.csv.gz")

    # 2. Hyperparameters
    learning_rate = 1e-2
    num_epochs = 10
    batch_size = 256
    hidden_dim = 64
    layer_list = [1, 3, 5, 7, 9]

    # 3. Train for each number of layers and evaluate on test set
    test_mses = []

    for L in layer_list:
        print(f"\n=== Training model with {L} layer(s) ===")
        mse_test, _ = train_one_model(
            X_train,
            y_train,
            X_test,
            y_test,
            lr=learning_rate,
            num_layers=L,
            num_epochs=num_epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            verbose=True,
        )
        test_mses.append(mse_test)

    # 4. plot test MSE vs number of layers
    plt.figure(figsize=(6, 4))
    plt.plot(layer_list, test_mses, marker="o")
    plt.xlabel("Number of layers")
    plt.ylabel("Test MSE (on data after 2020)")
    plt.title(f"Test MSE vs #Layers (lr={learning_rate}, epochs={num_epochs})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("test_mse_vs_layers.png", dpi=150)
    print("\nSaved plot to test_mse_vs_layers.png")

