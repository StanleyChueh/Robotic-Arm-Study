"""
Implementation of DP_BC. 
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.value_nets as ValueNets
import robomimic.models.vae_nets as VAENets

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.loss_utils as LossUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.diffusion_nets import DiffusionNetwork

@register_algo_factory_func("dp_bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the DP_BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    # only one variant of TD3_BC for now
    return DP_BC, {}

class DP_BC(PolicyAlgo):
    def __init__(self, **kwargs):
        print("🚀 DEBUG: Initializing DP_BC...")

        # ✅ Explicitly extract `algo_config`
        self.algo_config = kwargs.get("algo_config", None)
        if self.algo_config is None:
            raise AttributeError("🚨 ERROR: `algo_config` is missing in DP_BC!")
        
        # ✅ Set device **before** using it
        self.device = kwargs.get("device", torch.device("cpu"))  # Default to CPU if not provided

        print(f"✅ DEBUG: Using device: {self.device}")

        print(f"✅ DEBUG: self.algo_config.diffusion = {self.algo_config.diffusion}")

        # ✅ Assign diffusion settings **before** calling `PolicyAlgo.__init__`
        self.diffusion_steps = self.algo_config.diffusion.get("steps", 100)
        self.noise_schedule = self.algo_config.diffusion.get("noise_schedule", "cosine")

        # Convert noise schedule string into an actual tensor
        if isinstance(self.noise_schedule, str):
            if self.noise_schedule == "cosine":
                # Example: Generate a cosine noise schedule as a tensor
                timesteps = torch.linspace(0, 1, self.diffusion_steps, device=self.device)
                self.noise_schedule = torch.cos((timesteps + 0.008) / 1.008 * (torch.pi / 2)) ** 2
            elif self.noise_schedule == "linear":
                # Example: Generate a simple linear schedule
                self.noise_schedule = torch.linspace(0, 1, self.diffusion_steps, device=self.device)
            else:
                raise ValueError(f"🚨 ERROR: Unknown noise schedule '{self.noise_schedule}'!")

        self.noise_schedule = self.noise_schedule.to(self.device)
        print(f"✅ DEBUG: Converted noise_schedule to tensor: {self.noise_schedule.shape}, device={self.noise_schedule.device}")

        print(f"🚀 DEBUG: diffusion_steps = {self.diffusion_steps}")
        print(f"🚀 DEBUG: noise_schedule = {self.noise_schedule}")

        # Now, call the parent constructor **after assigning diffusion_steps**
        PolicyAlgo.__init__(self, **kwargs)

        # Now safely call `_create_networks()`
        self._create_networks()

    def _create_networks(self):
        """
        Create the networks for Diffusion Policy.
        """
        print(f"🚀 DEBUG: Creating networks with diffusion_steps = {self.diffusion_steps}")

        if not hasattr(self, "diffusion_steps"):
            raise AttributeError("🚨 ERROR: `diffusion_steps` is missing before network creation!")

        self.nets = nn.ModuleDict()

        obs_dim = sum(v[0] if isinstance(v, (list, tuple)) else v for v in self.obs_shapes.values())

        # Create the policy network
        self.nets["denoising_network"] = DiffusionNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            diffusion_steps=self.diffusion_steps,
            noise_schedule=self.noise_schedule,
        ).to(self.device)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Exactly the same as BCQ.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = {
            "obs": batch["obs"],  
            "actions": batch["actions"]
        }

        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)

            obs = batch["obs"]  # Extract observations
            actions = batch["actions"]  # Extract actions from dataset

            # Sample a random noise level
            t = torch.randint(0, self.diffusion_steps, (actions.shape[0],), device=self.device)

            # Add noise to actions
            noise = torch.randn_like(actions)
            noisy_actions = self._add_noise(actions, noise, t)

            # Predict noise instead of actions
            # Separate state and image observations
            state_obs = []
            image_obs = []

            for k in sorted(obs.keys()):
                if len(obs[k].shape) == 5:  # ✅ Only select true image tensors (B, T, C, H, W)
                    image_obs.append(obs[k])
                else:  # ✅ State tensors (1D, 2D, or 3D)
                    state_obs.append(obs[k])

            # Concatenate only state observations
            state_obs = [obs[k].to(self.device) for k in sorted(obs.keys()) if len(obs[k].shape) <= 2] 
            for img in image_obs:
                print(f"DEBUG: Image observation shape = {img.shape}")

            obs_tensor = torch.cat(state_obs, dim=-1) if state_obs else torch.zeros((actions.shape[0], 1), device=self.device)

            if image_obs:
                # Ensure all image tensors have the same shape
                image_obs = [img if img.ndim == 4 else img.unsqueeze(1) for img in image_obs]
                image_tensor = torch.cat(image_obs, dim=1)  # Concatenate along channel dimension
            else:
                image_tensor = None

            # Pass both state and image tensors into the denoising network
            if image_tensor is not None:
                predicted_noise = self.nets["denoising_network"](obs_tensor, image_tensor, noisy_actions, t)
            else:
                predicted_noise = self.nets["denoising_network"](obs_tensor, noisy_actions, t)

            # Compute loss: MSE between predicted noise and actual noise
            loss = F.mse_loss(predicted_noise, noise)
            info["loss"] = loss.item()

            if not validate:
                # Backpropagation
                self.optimizers["denoising_network"].zero_grad()
                loss.backward()
                self.optimizers["denoising_network"].step()

        return info

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        assert not self.nets.training
        return self._sample_from_diffusion(obs_dict)

    def _sample_from_diffusion(self, obs):
        """
        Sample action using iterative diffusion process.
        """
        action = torch.randn((obs.shape[0], self.ac_dim), device=self.device)

        for t in reversed(range(self.diffusion_steps)):
            predicted_noise = self.nets["denoising_network"](obs, action, t)
            action = self._remove_noise(action, predicted_noise, t)

            # Add stochastic noise to maintain the diffusion distribution
            if t > 0:
                noise = torch.randn_like(action)
                action += noise * self._get_beta(t)

        return action

    def _get_alpha(self, t):
        """
        Compute alpha value for the noise schedule.
        """
        alpha_t = torch.exp(-0.5 * torch.cumsum(self.noise_schedule.to(t.device), dim=0))
        return alpha_t[t]  # ✅ Ensure correct indexing

    def _add_noise(self, x, noise, t):
        """
        Add noise to the actions based on the noise schedule.

        Args:
            x (torch.Tensor): Clean action tensor, shape [batch_size, action_dim]
            noise (torch.Tensor): Random noise, same shape as x
            t (torch.Tensor): Time steps, shape [batch_size]

        Returns:
            torch.Tensor: Noisy actions
        """
        alpha_t = self._get_alpha(t).view(-1, 1)  # ✅ Ensure broadcasting
        return alpha_t * x + (1 - alpha_t) * noise

    def _remove_noise(self, x, predicted_noise, t):
        """
        Remove noise to denoise the action.
        """
        alpha_t = self._get_alpha(t)
        return (x - (1 - alpha_t) * predicted_noise) / alpha_t
    
    def _get_alpha(self, t):
        """
        Compute alpha value for the noise schedule.
        """
        return torch.exp(-0.5 * torch.cumsum(self.noise_schedule.to(t.device), dim=0)[t])

    def _get_beta(self, t):
        """
        Compute beta value for the noise schedule.
        """
        return self.algo_config.diffusion["beta_start"] + (self.algo_config.diffusion["beta_end"] - self.algo_config.diffusion["beta_start"]) * (t / self.diffusion_steps)

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        loss_log = OrderedDict()
        loss_log["Loss"] = info["loss"]
        return loss_log
