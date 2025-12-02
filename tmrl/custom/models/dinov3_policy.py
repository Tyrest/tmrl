import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg
import torchvision.transforms as T

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

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
        if x.shape[1] != 3:
            raise ValueError(f"Expected input with 3 channels, got {x.shape[1]} channels.")
        
        # Resize and Normalize
        x = self.resize(x)
        x = self.normalize(x)
        
        with torch.no_grad():
            features = self.backbone(x) # (B, Dim)
            
        return features

class DinoV3FeatureActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        print("Using DinoV3FeatureActorCritic model (Features Input).")
        self.actor = SquashedGaussianDinoV3FeatureActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = DinoV3FeatureQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = DinoV3FeatureQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()

class SquashedGaussianDinoV3FeatureActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        
        # We assume features are passed, so we don't need encoder.
        # We need to know feature dim. 
        # DINOv3 tiny output dim is 384.
        # Ideally we pass this or infer it.
        # For now hardcode or use dummy encoder to get dim?
        # Let's hardcode 768 for convnext_tiny.
        self.feature_dim = 768 * cfg.IMG_HIST_LEN
        
        self.mlp_input_features = self.feature_dim + 9
        self.mlp = mlp([self.mlp_input_features] + list(hidden_sizes), activation)
        
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, features, act1, act2 = obs
        # features: (B, Dim)
        
        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        
        mlp_in = torch.cat((speed, gear, rpm, features, act1, act2), -1)
        net_out = self.mlp(mlp_in)
        
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()

class DinoV3FeatureQFunction(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.feature_dim = 768 * cfg.IMG_HIST_LEN
        act_dim = action_space.shape[0]
        self.mlp_input_features = self.feature_dim + 9 + act_dim
        self.mlp = mlp([self.mlp_input_features] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        speed, gear, rpm, features, act1, act2 = obs
        
        if features.dim() > 2:
            features = features.flatten(start_dim=1)

        mlp_in = torch.cat((speed, gear, rpm, features, act1, act2, act), -1)
        q = self.mlp(mlp_in)
        return torch.squeeze(q, -1)

