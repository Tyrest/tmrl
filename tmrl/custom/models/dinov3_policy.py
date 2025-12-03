import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg


LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

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
