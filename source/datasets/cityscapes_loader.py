import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CityscapesDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((512, 1024)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        img_dir = os.path.join(root, "leftImg8bit", split)
        lbl_dir = os.path.join(root, "gtFine", split)

        self.files = []
        for city in os.listdir(img_dir):
            img_folder = os.path.join(img_dir, city)
            lbl_folder = os.path.join(lbl_dir, city)
            for fname in os.listdir(img_folder):
                if fname.endswith("_leftImg8bit.png"):
                    lbl_name = fname.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    self.files.append({
                        "img": os.path.join(img_folder, fname),
                        "lbl": os.path.join(lbl_folder, lbl_name)
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = self.files[idx]
        img = Image.open(sample["img"]).convert("RGB")
        lbl = Image.open(sample["lbl"])
        if self.transform:
            img = self.transform(img)
        return img, lbl
