import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_transform_compose() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))] # Normalized to [-1, 1]
    )


def get_mnist_dataset(
    transform: transforms.Compose,
    is_train: bool,
    root: str = "./data",
    download: bool = True,
) -> torchvision.datasets.MNIST:
    return torchvision.datasets.MNIST(
        root=root, train=is_train, download=download, transform=transform
    )


def get_mnist_data_loader(
    batch_size: int,
    shuffle: bool,
    is_train: bool,
    root: str = "./data",
    download: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset=get_mnist_dataset(
            transform=get_transform_compose(), is_train=is_train, root=root, download=download
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def show_image(image: torch.Tensor):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()
    return plt


if __name__ == "__main__":
    # train_loader = get_mnist_data_loader(batch_size=64, shuffle=True, is_train=True)
    dataset = get_mnist_dataset(
        transform=get_transform_compose(), is_train=True, root="./data", download=True
    )
    print(f'dataset length: {len(dataset)}')
    number = 10
    assert len(dataset) >= number
    for i in range(number):
        image, label = dataset[i]
        show_image(image)
        print(f'image shape: {image.shape}')
