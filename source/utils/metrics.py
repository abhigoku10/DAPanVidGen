import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from scipy import linalg
from transformers import CLIPProcessor, CLIPModel

def clip_score(video_frames, prompt, device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=[prompt], images=video_frames, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image 
    score = logits_per_image.mean().item()
    return score

def inception_score(video_frames, device="cuda", splits=10):
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    preds = []
    for frame in video_frames:
        frame = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = F.softmax(inception(frame), dim=1)
        preds.append(pred.numpy())

    preds = np.concatenate(preds, axis=0)
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds)//splits): (i+1) * (len(preds)//splits), :]
        kl = part * (np.log(part) - np.log(py))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores))

def fid_score(real_features, gen_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def fvd_score(real_videos, gen_videos, i3d_model, device="cuda"):
    i3d_model.eval()
    with torch.no_grad():
        real_feat = i3d_model(real_videos.to(device)).numpy()
        gen_feat = i3d_model(gen_videos.to(device)).numpy()

    mu1, sigma1 = np.mean(real_feat, axis=0), np.cov(real_feat, rowvar=False)
    mu2, sigma2 = np.mean(gen_feat, axis=0), np.cov(gen_feat, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fvd)
