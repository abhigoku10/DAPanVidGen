import torch
from diffusers import DiffusionPipeline

def get_weighted_text_embeddings(pipe: DiffusionPipeline, prompt, uncond_prompt=None,
                                 max_embeddings_multiples=3, no_boseos_middle=False):
    text_embeddings, uncond_embeddings = pipe.text_encoder(
        pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(pipe.device)
    )[0], None
    if uncond_prompt is not None:
        uncond_embeddings = pipe.text_encoder(
            pipe.tokenizer(uncond_prompt, return_tensors="pt").input_ids.to(pipe.device)
        )[0]
    return text_embeddings, uncond_embeddings

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
