from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.model import CNN
from src.dataset import (
    ProjectDataset,
    base_train_transform,
    strong_train_transform
)


def compute_class_weights(train_df, label_map):
    """
    Compute class weights from the training portion of one fold.

    Weight for class c:

        w_c = N / (K * N_c)

    where:
        N   = total number of training images
        K   = number of classes
        N_c = number of training images belonging to class c
    """

    weights = torch.zeros(
        len(label_map),
        dtype=torch.float32
    )

    for label, class_index in label_map.items():

        count = (train_df["dx"] == label).sum()

        weights[class_index] = (
            len(train_df)
            /
            (len(label_map) * count)
        )

    return weights


def train_cross_validation(
    crossval_df,
    label_map,
    n_splits=5,
    epochs=50,
    batch_size=32,
    learning_rate=0.0001,
    model_dir="models",
    device=None
):
    """
    Train the CNN using lesion-level GroupKFold cross-validation.

    A new CNN is initialized for each fold.

    The model achieving the highest validation balanced accuracy
    is saved for each fold.
    """

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # MODEL DIRECTORY
    # --------------------------------------------------------

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # GROUP CROSS-VALIDATION
    # --------------------------------------------------------

    group_kfold = GroupKFold(
        n_splits=n_splits
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        group_kfold.split(
            crossval_df,
            groups=crossval_df["lesion_id"]
        ),
        start=1
    ):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}/{n_splits}")
        print("=" * 60)

        # ----------------------------------------------------
        # TRAIN / VALIDATION DATAFRAMES
        # ----------------------------------------------------

        train_df = (
            crossval_df
            .iloc[train_idx]
            .reset_index(drop=True)
        )

        val_df = (
            crossval_df
            .iloc[val_idx]
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # DATASETS
        # ----------------------------------------------------

        train_dataset = ProjectDataset(
            train_df,
            label_map,
            base_transform=base_train_transform,
            strong_transform=strong_train_transform
        )

        val_dataset = ProjectDataset(
            val_df,
            label_map
        )

        # ----------------------------------------------------
        # DATA LOADERS
        # ----------------------------------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print(
            f"Training images: {len(train_dataset)}"
        )

        print(
            f"Validation images: {len(val_dataset)}"
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = CNN().to(device)

        # ----------------------------------------------------
        # CLASS WEIGHTS
        # ----------------------------------------------------

        weights = compute_class_weights(
            train_df,
            label_map
        ).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=weights
        )

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        # ----------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------

        best_balanced_accuracy = 0.0

        best_model_path = (
            model_dir
            /
            f"best_model_fold_{fold}.pth"
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        for epoch in range(epochs):

            model.train()

            # IMPORTANT:
            # these values must be reset at every epoch

            train_loss_sum = 0.0
            train_weight_sum = 0.0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(images)

                loss = criterion(
                    logits,
                    labels
                )

                loss.backward()

                optimizer.step()

                # Weighted epoch loss

                batch_weights = weights[labels]

                train_loss_sum += (
                    loss.item()
                    *
                    batch_weights.sum().item()
                )

                train_weight_sum += (
                    batch_weights.sum().item()
                )

            train_loss = (
                train_loss_sum
                /
                train_weight_sum
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            model.eval()

            val_loss_sum = 0.0
            val_weight_sum = 0.0

            all_labels = []
            all_predictions = []

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)

                    loss = criterion(
                        logits,
                        labels
                    )

                    batch_weights = weights[labels]

                    val_loss_sum += (
                        loss.item()
                        *
                        batch_weights.sum().item()
                    )

                    val_weight_sum += (
                        batch_weights.sum().item()
                    )

                    predictions = torch.argmax(
                        logits,
                        dim=1
                    )

                    all_labels.extend(
                        labels
                        .cpu()
                        .numpy()
                    )

                    all_predictions.extend(
                        predictions
                        .cpu()
                        .numpy()
                    )

            val_loss = (
                val_loss_sum
                /
                val_weight_sum
            )

            # ------------------------------------------------
            # VALIDATION METRICS
            # ------------------------------------------------

            balanced_accuracy = (
                balanced_accuracy_score(
                    all_labels,
                    all_predictions
                )
            )

            macro_f1 = f1_score(
                all_labels,
                all_predictions,
                average="macro"
            )

            # ------------------------------------------------
            # SAVE BEST MODEL
            # ------------------------------------------------

            if (
                balanced_accuracy
                >
                best_balanced_accuracy
            ):

                best_balanced_accuracy = (
                    balanced_accuracy
                )

                torch.save(
                    model.state_dict(),
                    best_model_path
                )

            # ------------------------------------------------
            # EPOCH OUTPUT
            # ------------------------------------------------

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Balanced accuracy: "
                f"{balanced_accuracy:.4f} | "
                f"Macro F1: {macro_f1:.4f}"
            )

        # ----------------------------------------------------
        # FOLD RESULT
        # ----------------------------------------------------

        fold_results.append({
            "fold": fold,
            "best_balanced_accuracy":
                best_balanced_accuracy,
            "model_path":
                str(best_model_path)
        })

        print(
            f"\nBest balanced accuracy "
            f"for fold {fold}: "
            f"{best_balanced_accuracy:.4f}"
        )

    return fold_results