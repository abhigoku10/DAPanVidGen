import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class PanoramaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform or T.Compose([
            T.Resize((512, 1024)),
            T.ToTensor()
        ])
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        prompt = "panoramic scene"
        return img, prompt
