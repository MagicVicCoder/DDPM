import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                           glob.glob(os.path.join(root_dir, "*.jpeg")) + \
                           glob.glob(os.path.join(root_dir, "*.png"))
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 64, 64)
