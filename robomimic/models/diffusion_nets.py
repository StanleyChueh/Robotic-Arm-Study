import torch
import torch.nn as nn

class DiffusionNetwork(nn.Module):
    def __init__(self, obs_shapes, ac_dim, diffusion_steps, noise_schedule, use_images=False):
        super(DiffusionNetwork, self).__init__()
        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.use_images = use_images

        obs_dim = sum(v[0] if isinstance(v, (list, tuple)) else int(v) for v in obs_shapes.values())

        if self.use_images:
            # ✅ Define CNN-based encoder for images
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.image_feature_dim = 64 * 21 * 21  # Adjusted feature size
            obs_dim += self.image_feature_dim  

        # ✅ Define the MLP for diffusion model
        self.model = nn.Sequential(
            nn.Linear(obs_dim + ac_dim + 1, 1024),  # Include `t` (1 extra dim)
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, ac_dim)
        )

    def forward(self, obs, actions, t, image=None):
        batch_size = obs.shape[0]

        # ✅ Flatten obs explicitly to [batch_size, obs_dim]
        obs = obs.view(batch_size, -1)

        # ✅ Correctly handle actions shape (should be [batch_size, ac_dim])
        if actions.dim() == 3:
            actions = actions[:, -1, :]  # select last timestep
        elif actions.dim() > 3:
            actions = actions.reshape(batch_size, -1)
            if actions.shape[1] != self.ac_dim:
                actions = actions[:, :self.ac_dim]  # truncate if necessary

        # Ensure actions dimension is explicitly ac_dim (7 in your config)
        actions = actions[:, :self.ac_dim]

        # ✅ Correct handling for t (time-step embedding), should be [batch_size, 1]
        if t.dim() > 2:
            t = t.view(batch_size, -1)
        t = t[:, :1]

        # ✅ Correct handling for images
        if self.use_images and image is not None:
            # Adjust from [batch_size, H, W, C] or [batch_size, 1, C, H, W] → [batch_size, C, H, W]
            if image.dim() == 5:
                image = image[:, -1]  # [batch_size, C, H, W]
            if image.shape[-1] == 3:  # channels last → channels first
                image = image.permute(0, 3, 1, 2)  # [B,H,W,C] → [B,C,H,W]

            # ✅ Use image_encoder to encode images
            image_features = self.image_encoder(image)

            # Ensure batch dimension matches
            assert image_features.shape[0] == batch_size

            obs = torch.cat([obs, image_features], dim=-1)

        # ✅ Dynamically compute expected observation dimension
        expected_obs_dim = sum(
            v[0] if isinstance(v, (list, tuple)) else int(v)
            for v in self.obs_shapes.values()
        )
        if self.use_images:
            expected_obs_dim += self.image_feature_dim

        # Adjust obs if mismatch
        if obs.shape[1] != expected_obs_dim:
            if obs.shape[1] > expected_obs_dim:
                obs = obs[:, :expected_obs_dim]
            else:
                padding = torch.zeros((batch_size, expected_obs_dim - obs.shape[1]), device=obs.device)
                obs = torch.cat([obs, padding], dim=-1)

        # Concatenate obs, actions, t
        x = torch.cat([obs, actions, t], dim=-1)

        # ✅ Final sanity-check before model input
        expected_dim = self.model[0].in_features
        if x.shape[1] != expected_dim:
            raise ValueError(f"Final input shape mismatch: Expected {expected_dim}, got {x.shape[1]}")

        return self.model(x)
