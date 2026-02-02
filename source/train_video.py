import torch
import yaml
from torch.utils.data import DataLoader
from datasets.kitti360_loader import Kitti360Dataset
from models.video_diffusion import PanoramicVideoDiffusion
from utils.logger import Logger


def train_video(config):
    dataset = Kitti360Dataset(config["data_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    model = PanoramicVideoDiffusion(config, vidgen_checkpoint=config["vidgen_checkpoint"]).to("cuda")
    for param in model.image_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["lr"]
    )
    logger = Logger()
    for epoch in range(config["epochs"]):
        for batch in dataloader:
            batch = [x.to("cuda") for x in batch]
            loss, metrics = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.log({"epoch": epoch, "loss": loss.item(), **metrics})
        torch.save(model.state_dict(), f"checkpoints/panoramic_model_stage2_{epoch}.pt")

if __name__ == "__main__":
    config = yaml.safe_load(open("configs.yaml"))
    train_video(config)

