import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class PokemonDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (image_path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_split(data_dir, val_ratio=0.2, seed=42):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    train_samples, val_samples = [], []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        images = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        label = class_to_idx[cls]
        train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=seed)
        train_samples += [(img, label) for img in train_imgs]
        val_samples += [(img, label) for img in val_imgs]

    return train_samples, val_samples, classes


def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size=32, img_size=224, val_ratio=0.2, num_workers=4):
    train_samples, val_samples, classes = load_split(data_dir, val_ratio)
    train_transform, val_transform = get_transforms(img_size)

    train_loader = DataLoader(
        PokemonDataset(train_samples, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        PokemonDataset(val_samples, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, classes
