import open_clip
import torch
from PIL import Image
import torch.nn.functional as F
import lpips
import numpy as np

class Evaluator:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        self.device = device
        # CLIP Setup
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip_model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # LPIPS Setup
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
    
    @torch.no_grad()
    def compute_clip_score(self, image_path, text):
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer([text]).to(self.device)
        
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text_tokens)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        score = torch.sum(image_features * text_features, dim=-1).item()
        return score

    @torch.no_grad()
    def compute_lpips(self, image_path1, image_path2):
        img1 = self.load_image_for_lpips(image_path1)
        img2 = self.load_image_for_lpips(image_path2)
        return self.lpips_model(img1, img2).item()

    def load_image_for_lpips(self, path):
        img = Image.open(path).convert('RGB').resize((512, 512))
        img = np.array(img).transpose(2, 0, 1) # HWC to CHW
        img = (img / 127.5) - 1.0 # Normalize to [-1, 1]
        return torch.from_numpy(img).float().unsqueeze(0).to(self.device)
