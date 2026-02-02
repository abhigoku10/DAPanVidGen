import torch
import numpy as np
import yaml
import argparse
from torchvision import transforms as T
from PIL import Image
from models.video_diffusion import PanoramicVideoDiffusion
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, logging
from models.blend_extend_pipeline import StableDiffusionBlendExtendPipeline
from models.video_diffusion import PanoramicVideoDiffusion


def get_views(height, width, window_size=64, stride=8):
    h, w = height // 8, width // 8
    num_h = (h - window_size) // stride + 1
    num_w = (w - window_size) // stride + 1
    views = []
    for i in range(num_h * num_w):
        hs, he = int((i // num_w) * stride), int((i // num_w) * stride) + window_size
        ws, we = int((i % num_w) * stride), int((i % num_w) * stride) + window_size
        views.append((hs, he, ws, we))
    return views


class MultiDiffusion(torch.nn.Module):
    def __init__(self, device, model_key="stabilityai/stable-diffusion-2-base"):
        super().__init__()
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

    @torch.no_grad()
    def encode_text(self, prompts, negatives):
        if isinstance(prompts, str): prompts = [prompts]
        if isinstance(negatives, str): negatives = [negatives]
        text_inputs = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt")
        text_embeds = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        neg_inputs = self.tokenizer(negatives, padding="max_length", max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt")
        neg_embeds = self.text_encoder(neg_inputs.input_ids.to(self.device))[0]
        return torch.cat([neg_embeds, text_embeds])

    @torch.no_grad()
    def decode(self, latents):
        latents = latents / 0.18215
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def generate_panorama(self, prompt, negative="", height=512, width=2048, steps=50, guidance=7.5):
        embeds = self.encode_text(prompt, negative)
        latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        self.scheduler.set_timesteps(steps)

        with torch.autocast("cuda"):
            for t in self.scheduler.timesteps:
                latent_in = torch.cat([latents] * 2)
                noise_pred = self.unet(latent_in, t, encoder_hidden_states=embeds)["sample"]
                uncond, cond = noise_pred.chunk(2)
                guided = uncond + guidance * (cond - uncond)
                latents = self.scheduler.step(guided, t, latents)["prev_sample"]

        img = self.decode(latents)
        return T.ToPILImage()(img[0])

    @torch.no_grad()
    def generate_masked(self, masks, prompts, negatives, height=512, width=2048, steps=50,
                        guidance=7.5, bootstrapping=20):
        embeds = self.encode_text(prompts, negatives)
        latents = torch.randn((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        noise = latents.clone().repeat(len(prompts) - 1, 1, 1, 1)
        views = get_views(height, width)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)
        self.scheduler.set_timesteps(steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                count.zero_(); value.zero_()
                for hs, he, ws, we in views:
                    mv = masks[:, :, hs:he, ws:we]
                    lv = latents[:, :, hs:he, ws:we].repeat(len(prompts), 1, 1, 1)
                    if i < bootstrapping:
                        bg = torch.randn_like(noise[:, :, hs:he, ws:we])
                        lv[1:] = lv[1:] * mv[1:] + bg * (1 - mv[1:])
                    model_in = torch.cat([lv] * 2)
                    noise_pred = self.unet(model_in, t, encoder_hidden_states=embeds)["sample"]
                    uncond, cond = noise_pred.chunk(2)
                    guided = uncond + guidance * (cond - uncond)
                    denoised = self.scheduler.step(guided, t, lv)["prev_sample"]
                    value[:, :, hs:he, ws:we] += (denoised * mv).sum(dim=0, keepdims=True)
                    count[:, :, hs:he, ws:we] += mv.sum(dim=0, keepdims=True)
                latents = torch.where(count > 0, value / count, value)

        img = self.decode(latents)
        return T.ToPILImage()(img[0])

def run_inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1: Generate panoramic image latent with LoRA + special token
    pipe = StableDiffusionBlendExtendPipeline.from_pretrained(
        config.get("model_key", "stabilityai/stable-diffusion-2-base"),
        torch_dtype=torch.float16
    ).to(device)

    # LoRA weights
    if "lora_path" in config:
        pipe.unet.load_attn_procs(config["lora_path"])

    pipe.vae.enable_tiling()
    pipe.scheduler.set_timesteps(config.get("steps", 50))

    pano_img = pipe(
        prompt=config["prompt"], 
        negative_prompt=config.get("negative", ""),
        height=config.get("H", 512),
        width=config.get("W", 2048),
        guidance_scale=config.get("guidance", 7.5),
        num_inference_steps=config.get("steps", 50),
    ).images[0]

    pano_img.save(config.get("outfile", "pano_stage1.png"))

    # Stage 2: Extend to video using temporal attention
    video_model = PanoramicVideoDiffusion(
        {"latent_dim": config.get("latent_dim", 256)},
        vidgen_checkpoint=config.get("vidgen_checkpoint")
    ).to(device)


    image_tensor = torch.tensor(np.array(pano_img)).permute(2, 0, 1).unsqueeze(0).to(device)
    image_latent = pipe.vae.encode(image_tensor / 255.0 * 2 - 1).latent_dist.sample() * 0.18215

    frames = video_model.run_inference(image_latent, steps=config.get("frames", 16))

    for idx, frame in enumerate(frames):
        frame.save(f"video_frame_{idx}.png")

    print(f"Generated {len(frames)} panoramic video frames.")
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    run_inference(config)
