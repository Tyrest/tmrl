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

class ResidualBlock(nn.Module):
    """
    Standard Impala/Impoola Residual Block.
    Structure: ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> Residual Add
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inputs

class ConvSequence(nn.Module):
    """
    ConvSequence block consisting of a convolution, max pooling, and two residual blocks.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

class ImpoolaCNN(nn.Module):
    """
    Impoola-CNN: An improved image encoder for Deep RL.
    Replaces the Flatten layer of Impala-CNN with Global Average Pooling (GAP).

    Input: (B, 3, 64, 64)
    """
    def __init__(self, input_channels=cfg.IMG_HIST_LEN, width_scale=1, output_dim=256):
        super().__init__()
        
        channels = (16 * width_scale, 32 * width_scale, 32 * width_scale)

        self.conv_seq0 = ConvSequence(input_channels, channels[0])
        self.conv_seq1 = ConvSequence(channels[0], channels[1])
        self.conv_seq2 = ConvSequence(channels[1], channels[2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channels[2], output_dim)
        self.relu = nn.ReLU()
        self.flat_features = output_dim

    def forward(self, x):
        x = self.conv_seq0(x)
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        
        x = self.gap(x)
        x = x.flatten(1)
        
        x = self.relu(x)
        z = self.linear(x)
        z = self.relu(z)
        
        return z

class ImpoolaCNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        print("Using ImpoolaCNNActorCritic model.")

        # build policy and value functions
        self.actor = SquashedGaussianImpoolaCNNActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = ImpoolaCNNQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = ImpoolaCNNQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


class SquashedGaussianImpoolaCNNActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        
        input_channels = cfg.IMG_HIST_LEN if cfg.GRAYSCALE else cfg.IMG_HIST_LEN * 3
        self.cnn = ImpoolaCNN(input_channels=input_channels)
        
        self.mlp_input_features = self.cnn.flat_features + 9
        self.mlp = mlp([self.mlp_input_features] + list(hidden_sizes), activation)
        
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act2 = obs
        
        if not cfg.GRAYSCALE:
            x = images.permute(0, 1, 4, 2, 3)
            B, Hist, C, H, W = x.shape
            x = x.reshape(B, Hist*C, H, W)
        else:
            x = images
        
        cnn_out = self.cnn(x)
        
        mlp_in = torch.cat((speed, gear, rpm, cnn_out, act1, act2), -1)
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


class ImpoolaCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        
        input_channels = cfg.IMG_HIST_LEN if cfg.GRAYSCALE else cfg.IMG_HIST_LEN * 3
        self.cnn = ImpoolaCNN(input_channels=input_channels)
        
        act_dim = action_space.shape[0]
        self.mlp_input_features = self.cnn.flat_features + 9 + act_dim
        self.mlp = mlp([self.mlp_input_features] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        
        if not cfg.GRAYSCALE:
            x = images.permute(0, 1, 4, 2, 3)
            B, Hist, C, H, W = x.shape
            x = x.reshape(B, Hist*C, H, W)
        else:
            x = images
        
        cnn_out = self.cnn(x)
        
        mlp_in = torch.cat((speed, gear, rpm, cnn_out, act1, act2, act), -1)
        q = self.mlp(mlp_in)
        
        return torch.squeeze(q, -1)
