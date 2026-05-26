import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class CombinedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim=256):
        super().__init__(observation_space, features_dim)


        # LIDAR BRANCH
        self.lidar_extractor = nn.Sequential(
            nn.Unflatten(1, (1, 180)),      # (B,1,180)

            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )

        # compute size automatically
        with th.no_grad():
            sample = th.zeros(1, 180)
            n_flatten = self.lidar_extractor(sample).shape[1]

        self.lidar_fc = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU()
        )

        # GOAL BRANCH
        self.goal_extractor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # FUSION
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),

            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        obs = obs.float()

        lidar = obs[:, :180]
        goal = obs[:, 180:]

        lidar_feat = self.lidar_fc(self.lidar_extractor(lidar))
        goal_feat = self.goal_extractor(goal)

        fused = th.cat([lidar_feat, goal_feat], dim=1)

        return self.fusion(fused)
