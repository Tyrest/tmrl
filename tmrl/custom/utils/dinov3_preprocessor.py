import torch
import numpy as np
from tmrl.custom.models.dinov3_policy import DinoV3Encoder
import tmrl.config.config_constants as cfg

class DinoV3Preprocessor:
    def __init__(self):
        self.encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_encoder(self):
        if self.encoder is None:
            print("Loading DINOv3 Encoder in Preprocessor...")
            input_channels = cfg.IMG_HIST_LEN if cfg.GRAYSCALE else cfg.IMG_HIST_LEN * 3
            self.encoder = DinoV3Encoder(input_channels=input_channels)
            self.encoder.to(self.device)
            self.encoder.eval()

    def __call__(self, obs):
        self._load_encoder()
        
        # obs is (speed, gear, rpm, images, *others)
        # images is numpy array (H, W, C) or (C, H, W) depending on pipeline
        # In tm_gym_interfaces.py, images are (img_hist_len, H, W, C) or similar?
        # Actually, obs_preprocessor_tm_act_in_obs converts images to float and normalizes.
        # But we want to replace that preprocessor or chain it.
        
        # Let's assume this replaces the standard preprocessor.
        # Standard preprocessor:
        # obs = (obs[0] / 1000.0, obs[1] / 10.0, obs[2] / 10000.0, grayscale_images, *obs[4:])
        
        speed = obs[0] / 1000.0
        gear = obs[1] / 10.0
        rpm = obs[2] / 10000.0
        images = obs[3] # (Hist, H, W, C) or (Hist, H, W) if grayscale
        
        # Prepare images for DINOv3
        # DINOv3 expects (B, 3, H, W)
        # images from gym are usually uint8 0-255 or float 0-1?
        # In TM2020Interface, grab_data_and_img returns img as (H, W, 3) BGR
        # And get_obs_rew_terminated_info appends to deque.
        # So images is (Hist, H, W, 3)
        
        # We need to stack history and channel dim
        # If Hist=1, (1, H, W, 3) -> (1, 3, H, W)
        # If Hist=4, (4, H, W, 3) -> (12, H, W) ? No, DINOv3 takes 3 channels usually.
        # But our DinoV3Encoder handles input_channels.
        
        # Convert to tensor
        x = torch.from_numpy(images).float().to(self.device)
        
        # Rearrange to (B, C, H, W) where B=Hist
        # images: (Hist, H, W, C)
        if len(x.shape) == 4: # (Hist, H, W, C)
            x = x.permute(0, 3, 1, 2) # (Hist, C, H, W)
            # x = x.reshape(-1, x.shape[2], x.shape[3]) # (Hist*C, H, W)
            # x = x.unsqueeze(0) # (1, Hist*C, H, W)
        elif len(x.shape) == 3: # (Hist, H, W) Grayscale
             x = x.unsqueeze(1) # (Hist, 1, H, W)
             # x = x.reshape(-1, x.shape[2], x.shape[3]) # (Hist, H, W)
             # x = x.unsqueeze(0) # (1, Hist, H, W)
             
        # Normalize 0-255 to 0-1 if needed?
        # TM2020Interface returns 0-255 uint8 usually.
        x = x / 255.0
        
        with torch.no_grad():
            features = self.encoder(x) # (Hist, Dim)
            
        features = features.flatten().cpu().numpy()
        
        return (speed, gear, rpm, features, *obs[4:])
