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
        # ✅ Ensure `t` has correct shape
        if t.dim() == 3:  # If t has multiple dimensions
            t = t[:, 0, 0].view(-1, 1)  # Extract one value per batch

        elif t.dim() == 1:
            t = t.view(-1, 1)  # Convert [batch_size] → [batch_size, 1]

        # ✅ Flatten `actions` if needed
        if actions.dim() > 2:
            batch_size = actions.shape[0]
            actions = actions.view(actions.shape[0], -1)  # Flatten to [batch_size, feature_dim]

        # ✅ Process images
        if self.use_images and image is not None:
            if image.dim() == 4:  # [batch_size, C, H, W]
                image = self.image_encoder(image)
            elif image.dim() == 5:  # [batch_size, frames, C, H, W]
                batch_size, frames, C, H, W = image.shape
                image = image.view(batch_size, -1)  # Flatten
            elif image.dim() == 6:
                batch_size, num_frames, num_channels, H, W, _ = image.shape
                image = image.view(batch_size, -1)  # Flatten
            else:
                raise ValueError(f"Unexpected image tensor shape: {image.shape}")

            obs = torch.cat([obs, image], dim=-1)  # Concatenate image features

        if obs.dim() == 3:
            obs = obs.view(obs.shape[0], -1)  # Flatten if needed

        if obs.dim() > 2:
            obs = obs.view(obs.shape[0], -1)  # Flatten obs to [batch_size, feature_dim]

        # ✅ Debugging prints
        print(f"DEBUG: obs shape = {obs.shape}, actions shape = {actions.shape}, t shape = {t.shape}")

        # ✅ Concatenate obs, actions, and t
        x = torch.cat([obs, actions, t], dim=-1)

        # Check expected input shape before passing through the network
        expected_dim = self.model[0].in_features
        if x.shape[1] != expected_dim:
            print(f"obs shape: {obs.shape}")
            print(f"image shape: {image.shape if image is not None else 'None'}")
            print(f"actions shape: {actions.shape}")
            print(f"t shape: {t.shape}")

            raise ValueError(f"Input shape mismatch: Expected {expected_dim}, got {x.shape[1]}")

        return self.model(x)
