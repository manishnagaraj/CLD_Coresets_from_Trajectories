# Compute_CLD_scores.py
import pickle
import numpy as np
import fire
import os
from tqdm import tqdm

def main(
    loss_path: str = 'Scores/Train_loss.pkl',
    dataset: str = 'CIFAR100',
    model_arch: str = 'resnet18',
    split_path: str = ''  # optional: path to ./Splits/<dataset>_<arch>_seed<seed>_split.pkl to map back to original indices
):
    ds = dataset.lower()
    if ds == 'cifar100':
        num_classes = 100
    elif ds == 'cifar10':
        num_classes = 10
    elif ds == 'imagenet':
        num_classes = 1000
    else:
        raise ValueError("Unsupported dataset. Use 'CIFAR10', 'CIFAR100', or 'ImageNet'.")

    with open(loss_path, 'rb') as f:
        loaded = pickle.load(f)

    # Expect lists per epoch
    tr_loss_list = loaded['tr_loss']      # list[E] of list/array[N_train]
    val_loss_list = loaded['val_loss']    # list[E] of list/array[N_val]
    tr_lbl_list  = loaded['tr_lbl']       # list[E] of list/array[N_train]
    val_lbl_list = loaded['val_lbl']      # list[E] of list/array[N_val]

    # stack into (E, N) arrays; be robust to lists
    train_losses = np.vstack([np.asarray(x, dtype=float) for x in tr_loss_list])  # (E, N_train)
    val_losses   = np.vstack([np.asarray(x, dtype=float) for x in val_loss_list]) # (E, N_val)

    # labels (take from first epoch; order is stable across epochs)
    train_labels = np.asarray(tr_lbl_list[0])
    val_labels   = np.asarray(val_lbl_list[0])

    E, N_train = train_losses.shape
    E2, N_val  = val_losses.shape
    assert E == E2, "Train and val losses must have same number of epochs"

    # diffs across epochs
    d_train = np.diff(train_losses, axis=0)  # (E-1, N_train)

    # optional mapping to original dataset indices
    orig_train_indices = None
    if split_path and os.path.isfile(split_path):
        with open(split_path, 'rb') as f:
            split = pickle.load(f)
        # split['train_indices'] should be length N_train
        if len(split.get('train_indices', [])) == N_train:
            orig_train_indices = np.asarray(split['train_indices'])
        else:
            print("[!] split_path provided but train_indices length does not match; ignoring.")

    CDL_sort = {}

    # pre-compute y standardization per class inside loop
    for i in tqdm(range(num_classes)):
        train_mask = (train_labels == i)
        val_mask   = (val_labels == i)

        # If a class is absent in train or val, skip cleanly
        if not np.any(train_mask) or not np.any(val_mask):
            CDL_sort[str(i)] = np.array([], dtype=int)
            continue

        # average val loss for class i per epoch -> (E,)
        avg_val_loss = val_losses[:, val_mask].mean(axis=1)
        d_avg_val = np.diff(avg_val_loss)  # (E-1,)

        # matrix of train diffs for class i: (E-1, M)
        X = d_train[:, train_mask]
        M = X.shape[1]

        # compute Pearson r for each column vs d_avg_val
        # r_j = corr(X[:, j], d_avg_val)
        # standardize
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0)
        y = d_avg_val
        y_mean = y.mean()
        y_std  = y.std()

        # avoid division by zero
        X_std_safe = np.where(X_std == 0, 1.0, X_std)
        y_std_safe = 1.0 if y_std == 0 else y_std

        Xz = (X - X_mean) / X_std_safe
        yz = (y - y_mean) / y_std_safe
        r = (Xz * yz[:, None]).mean(axis=0)  # (M,)

        # sort descending correlation
        order = np.argsort(r)[::-1]  # high to low

        # map back to train-subset indices
        train_idx_positions = np.flatnonzero(train_mask)  # positions in train subset
        sorted_positions = train_idx_positions[order]

        # optionally map to original dataset indices
        if orig_train_indices is not None:
            CDL_sort[str(i)] = orig_train_indices[sorted_positions]
        else:
            CDL_sort[str(i)] = sorted_positions

    os.makedirs('Scores', exist_ok=True)
    save_path = os.path.join('Scores', f'CLD_indices_sorted_{dataset}_{model_arch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(CDL_sort, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"CLD scores saved to {save_path}")

if __name__ == '__main__':
    fire.Fire(main)
