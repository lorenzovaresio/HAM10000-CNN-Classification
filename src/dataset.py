import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# DATA AUGMENTATION
# ============================================================

base_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])


strong_train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),

    transforms.ToTensor()
])


# Classes receiving stronger augmentation:
# akiec = 0
# bcc   = 1
# df    = 3
# vasc  = 6

STRONG_AUGMENTATION_CLASSES = {0, 1, 3, 6}


# ============================================================
# DATASET
# ============================================================

class ProjectDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 dermoscopic images.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the image and diagnostic label.

    label_map : dict
        Mapping from diagnostic labels to numerical class indices.

    base_transform : callable, optional
        Standard augmentation pipeline.

    strong_transform : callable, optional
        Stronger augmentation pipeline used for selected
        minority classes.
    """

    def __init__(
        self,
        df,
        label_map,
        base_transform=None,
        strong_transform=None
    ):
        self.df = df
        self.label_map = label_map
        self.base_transform = base_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image = self.df["image"].iloc[index]

        diagnostic_label = self.df["dx"].iloc[index]

        label = self.label_map[diagnostic_label]

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if (
            self.base_transform is not None
            and self.strong_transform is not None
        ):

            if label in STRONG_AUGMENTATION_CLASSES:
                image = self.strong_transform(image)
            else:
                image = self.base_transform(image)

        # ----------------------------------------------------
        # Validation / Test
        # ----------------------------------------------------

        else:

            image = torch.tensor(
                image,
                dtype=torch.float32
            )

            # H x W x C -> C x H x W
            image = image.permute(2, 0, 1)

            # RGB values: [0, 255] -> [0, 1]
            image = image / 255.0

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label