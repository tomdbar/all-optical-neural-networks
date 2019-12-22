import os
from enum import Enum

import torch
import torchvision


def mk_dir(export_dir):
    if not os.path.exists(export_dir):
            try:
                os.makedirs(export_dir)
                print('created dir: ', export_dir)
            except OSError as exc: # Guard against race condition
                 if exc.errno != exc.errno.EEXIST:
                    raise
            except Exception:
                pass
    else:
        print('dir already exists: ', export_dir)


class Dataset(Enum):
    MNIST = 0
    KMNIST = 1
    FASHION_MNIST = 2
    EMNIST = 3 # <-- Default URL for this dataset is currently offline (https://github.com/pytorch/vision/issues/1296).

def get_dataset_loaders(dataset=Dataset.MNIST,
                        train_batch=64,
                        test_batch=1000,
                        get_validation=False,
                        dir='./mnist_data/',
                        unroll_img=True,
                        max_value=1,
                        **kwargs):
    '''
    Generate the DataLoaders for various datasets.

    Args:
        dataset: The target dataset.
        train_batch: The training batch size.
        test_batch: The testing batch size.
        get_validation: Whether to return an additional third dataset for an unbiased test of the trained network.
        dir: Dataset directory.
        unroll_img: Whether the images should be unrolled into a vector (for MLP) or not (for Conv Net's).
        max_value: The maximum value to rescale the training data to.
        **kwargs: Other arguments passed to torch.utils.data.DataLoader constructor.

    Returns: A list of the training data loader, the test data loader and, optionally, the validation data loader.

    '''

    dataset = __find_dataset(dataset)

    transforms = [torchvision.transforms.ToTensor()]

    if unroll_img:

        class ReshapeTransform:
            def __init__(self, new_size):
                self.new_size = new_size

            def __call__(self, img):
                return img.view(img.size(0), -1)

        transforms.append(ReshapeTransform((-1,)))  # Reshape 28*28 array to vector.

    if max_value!=1:

        class RescaleTransform:
            def __init__(self, new_max):
                self.new_max = new_max

            def __call__(self, ft):
                return ft * self.new_max

        transforms.append(RescaleTransform(max_value))

    train_loader = torch.utils.data.DataLoader(
        dataset(dir, train=True, download=True,
                transform=torchvision.transforms.Compose(transforms)),
                batch_size=train_batch, shuffle=True, **kwargs)

    if not get_validation:

        test_loader = torch.utils.data.DataLoader(
            dataset(dir, train=False, download=True,
                    transform=torchvision.transforms.Compose(transforms)),
                    batch_size=test_batch, shuffle=True, **kwargs)

        loaders = [train_loader, test_loader]

    else:

        dataset = dataset(dir, train=False, download=True,
                          transform=torchvision.transforms.Compose(transforms))
        test_dataset, validation_dataset = torch.utils.data.random_split(dataset,[int(len(dataset)/2)]*2)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=True, **kwargs)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=test_batch, shuffle=True, **kwargs)

        loaders = [train_loader, test_loader, validation_loader]

    return loaders

def __find_dataset(dataset="mnist"):
    if dataset==Dataset.MNIST:
        ret = torchvision.datasets.MNIST
    elif dataset==Dataset.FASHION_MNIST:
        ret = torchvision.datasets.FashionMNIST
    elif dataset==Dataset.KMNIST:
        ret = torchvision.datasets.KMNIST
    elif dataset==Dataset.EMNIST:
        # Manually overwrite URL for this dataset (https://github.com/pytorch/vision/issues/1296).
        torchvision.datasets.EMNIST.url = 'https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download'
        ret = lambda *args, **kwargs: torchvision.datasets.EMNIST(split="balanced", *args, **kwargs)
    else:
        raise ValueError("{} is not a recognised dataset.  Acceptable values are {}".format(
            dataset,
            [d for d in Dataset]
        ))
    return ret