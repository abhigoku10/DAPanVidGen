import torch
import torch.nn as nn
from source.models.temporal_attention import RecurrentTemporalTransformer
from models.decoder import AdaptiveDecoder
from models.imgdiffusion import PanoramicImageDiffusion


class PanoramicVideoDiffusion(nn.Module):
    def __init__(self, config, image_checkpoint=None):
        super().__init__()
        self.latent_dim = config.get("latent_dim", 256)
        self.image_model = PanoramicImageDiffusion(config)
        if image_checkpoint:
            self.image_model.load_state_dict(torch.load(image_checkpoint))
        self.temporal_attn = RecurrentTemporalTransformer(latent_dim=self.latent_dim, heads=4)
        self.decoder = AdaptiveDecoder(latent_dim=self.latent_dim)
        self.lambda_motion = config.get("lambda_motion", 1.0)
        self.lambda_recon = config.get("lambda_recon", 1.0)

    def training_step(self, batch):
        frames, latents = batch 

        latents = latents.permute(1, 0, 2) 
        motion_latents = self.temporal_attn(latents)

        loss_motion = ((motion_latents - latents) ** 2).mean()

        recon = self.decoder(motion_latents.permute(1, 2, 0).unsqueeze(-1))
        loss_recon = ((recon - frames) ** 2).mean()

        loss = self.lambda_motion * loss_motion + self.lambda_recon * loss_recon
        return loss, {"loss_motion": loss_motion.item(), "loss_recon": loss_recon.item()}

    @torch.no_grad()
    def run_inference(self, prompt, steps=16):
        latent = self.image_model.generate_latent(prompt)

        latents = []
        current = latent.mean(dim=[2, 3])
        for _ in range(steps):
            current = current + torch.randn_like(current) * 0.05
            latents.append(current)
        latents = torch.stack(latents)

        motion_latents = self.temporal_attn(latents)

        frames = []
        for latent in motion_latents:
            latent = latent.unsqueeze(-1).unsqueeze(-1)
            frame = self.decoder(latent)
            frames.append(frame)
        return frames
