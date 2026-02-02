import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class Kitti360Dataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((512, 1024)),
            T.ToTensor()
        ])
        img_dir = os.path.join(root, "data_2d_raw", split)
        self.files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        caption = "A 360 degree view"
        return img, caption
