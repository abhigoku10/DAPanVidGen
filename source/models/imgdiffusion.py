import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig


class PanoramicImageDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)
        )

        lora_config = LoraConfig(
            r=8,                    
            lora_alpha=16,         
            target_modules=["0", "2"],
            lora_dropout=0.1,
            bias="none"
        )
        self.unet = get_peft_model(self.unet, lora_config)

        self.text_encoder = nn.Embedding(config["vocab_size"], config["embed_dim"])

        self.token_prompt = config.get("token_prompt", "A 360 degree view")
        self.special_token_id = config.get("special_token_id", 9999)
        self.text_encoder.weight.data[self.special_token_id] = torch.randn(config["embed_dim"])

        self.lambda_img = config.get("lambda_img", 1.0)

    def training_step(self, batch):
        images, captions = batch
        text_latent = self.text_encoder(captions)
        pred = self.unet(images + text_latent.unsqueeze(-1).unsqueeze(-1))
        loss = self.lambda_img * torch.abs(pred - images).mean()
        return loss

    def generate_latent(self, prompt):
        if self.token_prompt in prompt:
            tokens = torch.tensor([self.special_token_id]).unsqueeze(0)
        else:
            tokens = torch.randint(0, 100, (1, 10))
        text_latent = self.text_encoder(tokens)
        latent = torch.randn(1, 4, 64, 64) + text_latent.mean()
        return latent

    @classmethod
    def load_from_checkpoint(cls, path, config):
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model
