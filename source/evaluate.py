import torch
import numpy as np
import yaml
from utils.metrics import clip_score, inception_score, fvd_score, fid_score
from datasets.cityscapes_loader import CityscapesDataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from inference import run_inference 
from PIL import Image


def get_prompt_from_blip(image_path, device="cuda"):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CityscapesDataset(config["data_path"])

    scores = {"clip": [], "inception": [], "fvd": [], "fid": []}

    real_videos = []
    real_features = []
    for sample in dataset:
        real_videos.append(sample["video_tensor"])  
        real_features.append(sample["feature_vector"]) 

    real_videos = torch.stack(real_videos).to(device) 
    real_features = np.stack(real_features)         

    for sample in dataset:
        image_path = sample.get("image_path")
        prompt = sample.get("description")

        if not prompt or prompt.strip() == "":
            prompt = get_prompt_from_blip(image_path, device)

        config["prompt"] = prompt

        frames = run_inference(config)

        if isinstance(frames[0], Image.Image):
            frames_tensor = torch.stack([
                torch.tensor(np.array(f)).permute(2, 0, 1) for f in frames
            ]).float() / 255.0
        else:
            frames_tensor = frames

        scores["clip"].append(clip_score(frames, prompt, device=device))
        scores["inception"].append(inception_score(frames_tensor, device=device))
        scores["fvd"].append(fvd_score(real_videos, frames_tensor.unsqueeze(0), config["i3d_model"], device=device))
        scores["fid"].append(fid_score(real_features, frames_tensor.view(len(frames_tensor), -1).numpy()))

    results = {
        "CLIPScore": sum(scores["clip"]) / len(scores["clip"]),
        "Inception Score": sum(scores["inception"]) / len(scores["inception"]),
        "Frechet Video Distance": sum(scores["fvd"]) / len(scores["fvd"]),
        "Frechet Inception Distance": sum(scores["fid"]) / len(scores["fid"]),
    }

    print("\nEvaluation Results:")
    print("---------------------------------------------------")
    print(f"{'Metric':<30} {'Score':>10}")
    print("---------------------------------------------------")
    for k, v in results.items():
        print(f"{k:<30} {v:>10.2f}")
    print("---------------------------------------------------")

    return results


if __name__ == "__main__":
    config = yaml.safe_load(open("configs/eval.yaml"))
    evaluate(config)

