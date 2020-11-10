from pathlib import Path
from torch.utils.data import Dataloader
from torchvision import datasets, transforms


def get_dataloader(root_path, subset, transform, batch, workers):
    '''
    Returns a dataloader object
    -----------------------
    Args:
        - root_path (str)
        - subset (str)
            "train", "val", "test"
        - transform (torch transform obj)

    Returns: dataloader object
    -----------------------
    '''
    datafolder = datasets.ImageFolder(
        Path.joinpath(root_path, subset),
        transform,
    )
    if subset == "train":
        dataloader = Dataloader(
            datafolder,
            batch_size=batch,
            shuffle=True,
            num_workers=workers
        )
    else:
        dataloader = Dataloader(
            datafolder,
            batch_size=batch,
            shuffle=False,
            num_workers=workers
        )
    return dataloader


def get_transforms(rotation, scale, shear):
    '''
   Returns a transform object
    -----------------------
    Args:
        - subset (str)
            "train", "val", "test"

    Returns: transform object
    -----------------------
    '''
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=rotation, translate=None, scale=scale, shear=shear),
            transforms.RandomCrop(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    return train_transforms, test_transforms
