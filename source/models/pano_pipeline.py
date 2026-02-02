import torch
import torch.nn as nn
from diffusers import EulerAncestralDiscreteScheduler
from .pipeline_base import StableDiffusionBlendExtendPipeline
from .pipeline_base import get_weighted_text_embeddings, rescale_noise_cfg


class PanoramaTrainer(nn.Module):
    def __init__(self, model_id, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.pipe = StableDiffusionBlendExtendPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
        self.pipe.vae.enable_tiling()
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.device = device

    def training_step(self, imgs, prompts):
        imgs = imgs.to(self.device)
        latents = self.pipe.vae.encode(imgs * 2 - 1).latent_dist.sample() * 0.18215

        text_embeds, neg_embeds = get_weighted_text_embeddings(
            pipe=self.pipe,
            prompt=prompts,
            uncond_prompt=[""] * len(prompts),
            max_embeddings_multiples=3,
            no_boseos_middle=False
        )
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=self.device).long()
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text=noise_pred, guidance_rescale=0.7)
        loss = nn.MSELoss()(noise_pred, noise)
        return loss
