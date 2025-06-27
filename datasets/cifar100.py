import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

def load_dataset(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader