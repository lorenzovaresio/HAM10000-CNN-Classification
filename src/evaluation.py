from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

from src.model import CNN
from src.dataset import ProjectDataset


# ============================================================
# MODEL LOADING
# ============================================================

def load_models(
    n_models=5,
    model_dir="models",
    device=None
):
    """
    Load the trained CNN models obtained from cross-validation.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model_dir = Path(model_dir)

    models = []

    for fold in range(1, n_models + 1):

        model_path = (
            model_dir
            /
            f"best_model_fold_{fold}.pth"
        )

        model = CNN()

        state_dict = torch.load(
            model_path,
            map_location=device
        )

        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        models.append(model)

    return models


# ============================================================
# TEST DATA LOADER
# ============================================================

def create_test_loader(
    test_df,
    label_map,
    batch_size=32
):
    """
    Create a PyTorch DataLoader for the independent test set.
    """

    test_df = test_df.reset_index(drop=True)

    test_dataset = ProjectDataset(
        test_df,
        label_map
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


# ============================================================
# MULTICLASS EVALUATION
# ============================================================

def evaluate_models(
    models,
    test_loader,
    device=None
):
    """
    Evaluate all models on the independent test set.

    Returns
    -------
    results : list of dictionaries
        Performance metrics for every model.

    confusion_matrices : list
        Row-normalized confusion matrix for every model.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    results = []
    confusion_matrices = []

    for i, model in enumerate(models, start=1):

        model = model.to(device)
        model.eval()

        y_true = []
        y_pred = []

        with torch.inference_mode():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                y_true.extend(
                    labels.cpu().numpy()
                )

                y_pred.extend(
                    predictions.cpu().numpy()
                )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_true,
            y_pred,
            normalize="true"
        )

        confusion_matrices.append(cm)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        )

        macro_precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        results.append({
            "model": i,
            "accuracy": accuracy,
            "balanced_accuracy":
                balanced_accuracy,
            "macro_precision":
                macro_precision,
            "macro_recall":
                macro_recall,
            "macro_f1":
                macro_f1,
        })

    return results, confusion_matrices


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_results(results):
    """
    Print model-specific results and mean ± standard deviation.
    """

    for result in results:

        print(
            f"\n--- MODEL {result['model']} ---"
        )

        print(
            f"Accuracy: "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{result['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro Precision: "
            f"{result['macro_precision']:.4f}"
        )

        print(
            f"Macro Recall: "
            f"{result['macro_recall']:.4f}"
        )

        print(
            f"Macro F1: "
            f"{result['macro_f1']:.4f}"
        )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    print("\n==========================")
    print("MEAN ACROSS 5 MODELS")
    print("==========================")

    for metric in metric_names:

        values = np.array([
            result[metric]
            for result in results
        ])

        print(
            f"{metric}: "
            f"{values.mean():.4f} ± "
            f"{values.std():.4f}"
        )


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(
    confusion_matrices,
    label_map,
    save_path=None
):
    """
    Plot normalized confusion matrices for all models.
    """

    class_names = [
        label
        for label, index in sorted(
            label_map.items(),
            key=lambda item: item[1]
        )
    ]

    fig, axes = plt.subplots(
        1,
        len(confusion_matrices),
        figsize=(30, 5)
    )

    for i, cm in enumerate(
        confusion_matrices
    ):

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        display.plot(
            ax=axes[i],
            colorbar=False
        )

        axes[i].set_title(
            f"Model {i + 1}"
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# ROC: NEVI VS ALL OTHER LESIONS
# ============================================================

def plot_roc_nv(
    models,
    test_loader,
    label_map,
    device=None,
    save_path=None
):
    """
    ROC analysis for melanocytic nevi versus all other lesions.
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    nv_index = label_map["nv"]

    auc_values = []

    plt.figure(
        figsize=(8, 6)
    )

    for i, model in enumerate(
        models,
        start=1
    ):

        model = model.to(device)
        model.eval()

        y_true_nv = []
        y_score_nv = []

        with torch.inference_mode():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                nv_probabilities = (
                    probabilities[:, nv_index]
                )

                binary_labels = (
                    labels == nv_index
                ).int()

                y_true_nv.extend(
                    binary_labels
                    .cpu()
                    .numpy()
                )

                y_score_nv.extend(
                    nv_probabilities
                    .cpu()
                    .numpy()
                )

        auc = roc_auc_score(
            y_true_nv,
            y_score_nv
        )

        auc_values.append(auc)

        fpr, tpr, _ = roc_curve(
            y_true_nv,
            y_score_nv
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"Model {i} - "
                f"AUC = {auc:.3f}"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - nv vs all"
    )

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return auc_values


# ============================================================
# ROC: MALIGNANT VS NON-MALIGNANT
# ============================================================

def plot_roc_malignant(
    models,
    test_loader,
    label_map,
    device=None,
    save_path=None
):
    """
    ROC analysis for malignant versus non-malignant lesions.

    Malignant classes:
        - melanoma
        - basal cell carcinoma
        - actinic keratoses / intraepithelial carcinoma
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    mel_index = label_map["mel"]
    bcc_index = label_map["bcc"]
    akiec_index = label_map["akiec"]

    auc_values = []

    plt.figure(
        figsize=(8, 6)
    )

    for i, model in enumerate(
        models,
        start=1
    ):

        model = model.to(device)
        model.eval()

        y_true_malignant = []
        y_score_malignant = []

        with torch.inference_mode():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                # Probability assigned to any
                # malignant diagnostic class

                malignant_probabilities = (
                    probabilities[:, mel_index]
                    + probabilities[:, bcc_index]
                    + probabilities[:, akiec_index]
                )

                # 1 = malignant
                # 0 = non-malignant

                binary_labels = (
                    (labels == mel_index)
                    | (labels == bcc_index)
                    | (labels == akiec_index)
                ).int()

                y_true_malignant.extend(
                    binary_labels
                    .cpu()
                    .numpy()
                )

                y_score_malignant.extend(
                    malignant_probabilities
                    .cpu()
                    .numpy()
                )

        auc = roc_auc_score(
            y_true_malignant,
            y_score_malignant
        )

        auc_values.append(auc)

        fpr, tpr, _ = roc_curve(
            y_true_malignant,
            y_score_malignant
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"Model {i} - "
                f"AUC = {auc:.3f}"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Malignant vs non-malignant"
    )

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return auc_values