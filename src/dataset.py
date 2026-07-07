import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

def get_dataloaders(data_dir="data/flower_photos", batch_size=32, img_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes

    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names
