import torch
import yaml
from models.imgdiffusion import PanoramicImageDiffusion
from datasets.kitti360_loader import Kitti360Dataset
from utils.logger import Logger


def train_image(config):
    dataset = Kitti360Dataset(config["data_path"])
    model = PanoramicImageDiffusion(config).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    logger = Logger()

    for epoch in range(config["epochs"]):
        for batch in dataset:
            batch = [x.to("cuda") for x in batch]
            loss = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.log({"epoch": epoch, "loss": loss.item()})
        torch.save(model.state_dict(), f"checkpoints/panoramic_model_stage1_{epoch}.pt")


if __name__ == "__main__":
    config = yaml.safe_load(open("configs.yaml"))
    train_image(config)
