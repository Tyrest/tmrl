import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import tmrl.config.config_constants as cfg

class DinoV3Encoder(nn.Module):
    def __init__(self, input_channels=cfg.IMG_HIST_LEN):
        super().__init__()
        
        repo = "/home/tyler/Documents/CMU/dinov3"
        model_name = "dinov3_convnext_tiny"
        weights_path = "/home/tyler/Documents/CMU/tmrl/tmrl/custom/models/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
        
        print(f"Loading {model_name} from {repo}...")
        try:
            self.backbone = torch.hub.load(repo, model_name, source="local", weights=weights_path)
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Error loading DINOv3: {e}")
            raise e

        self.resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        # Determine output dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)
            self.backbone_dim = out.shape[1]
            print(f"DINOv3 output dim: {self.backbone_dim}")
        
        self.flat_features = self.backbone_dim

    def forward(self, x):
        # x comes in as (B, 3, H, W) or (B, 1, H, W)
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Resize and Normalize
        x = self.resize(x)
        x = self.normalize(x)
        
        with torch.no_grad():
            features = self.backbone(x) # (B, Dim)
            
        return features

class DinoV3Preprocessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = DinoV3Encoder().to(self.device)
        self.encoder.eval()
        print(f"DinoV3Preprocessor initialized on {self.device}")

    def __call__(self, obs):
        # obs is (speed, gear, rpm, images, *actions)
        # images is (Hist, H, W, 3) or (Hist, H, W)
        # We need to process the images and return features.
        
        speed = obs[0]
        gear = obs[1]
        rpm = obs[2]
        images = obs[3]
        actions = obs[4:]
        
        # Convert images to tensor
        # images: numpy array
        if len(images.shape) == 3: # (Hist, H, W) Grayscale
             x = torch.from_numpy(images).float().unsqueeze(1) # (Hist, 1, H, W)
             x = x.repeat(1, 3, 1, 1) # (Hist, 3, H, W)
        elif len(images.shape) == 4: # (Hist, H, W, 3) Color
             x = torch.from_numpy(images).float().permute(0, 3, 1, 2) # (Hist, 3, H, W)
        else:
            # If it's a single image (H, W, 3) or (H, W)
            if len(images.shape) == 2:
                x = torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
                x = x.repeat(1, 3, 1, 1)
            elif len(images.shape) == 3 and images.shape[2] == 3:
                x = torch.from_numpy(images).float().permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected images shape: {images.shape}")

        x = x.to(self.device)
        
        # Normalize to 0-1 if it's 0-255
        if x.max() > 1.0:
            x = x / 255.0
            
        with torch.no_grad():
            features = self.encoder(x) # (Hist, 768)
            
        # Flatten features: (Hist * 768)
        features_flat = features.cpu().numpy().flatten()
        
        return (speed, gear, rpm, features_flat, *actions)
