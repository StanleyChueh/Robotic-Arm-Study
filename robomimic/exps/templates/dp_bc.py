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
        PolicyAlgo.__init__(self, **kwargs)

        # Ensure `algo_config` exists before accessing attributes
        if not hasattr(self, "algo_config"):
            raise AttributeError("🚨 ERROR: `algo_config` is missing in DP_BC!")
        
        # Debugging: Print algo_config content
        print(f"🚀 DEBUG: self.algo_config = {self.algo_config}")
        
        # Diffusion, noise
        if not hasattr(self.algo_config, "diffusion"):
            raise AttributeError("❌ ERROR: `diffusion` is missing from `algo_config`. Check your config!")

        # Debugging: Print diffusion settings
        print(f"✅ DEBUG: self.algo_config.diffusion = {self.algo_config.diffusion}")

        self.diffusion_steps = self.algo_config.diffusion.get("steps", 100)  # Default to 100 if missing
        self.noise_schedule = self.algo_config.diffusion.get("noise_schedule", "cosine")

        print(f"🚀 DEBUG: diffusion_steps = {self.diffusion_steps}")  # Ensure it prints correctly
        print(f"🚀 DEBUG: noise_schedule = {self.noise_schedule}")  # Ensure this is correctly loaded

        # Create networks
        self._create_networks()

        # Define optimizer
        self.optimizers["denoising_network"] = TorchUtils.create_optimizer(
            self.algo_config.optim_params.policy, self.nets["denoising_network"]
        )

    def _create_networks(self):
        """
        Create the networks for Diffusion Policy.
        """
        print(f"🚀 DEBUG: Creating networks with diffusion_steps = {self.diffusion_steps}")

        if not hasattr(self, "diffusion_steps"):
            raise AttributeError("🚨 ERROR: `diffusion_steps` is missing before network creation!")

        self.nets = nn.ModuleDict()

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
            predicted_noise = self.nets["denoising_network"](obs, noisy_actions, t)

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

    def _add_noise(self, x, noise, t):
        """
        Add noise to the actions based on the noise schedule.
        """
        alpha_t = self._get_alpha(t)
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
        return torch.exp(-0.5 * torch.cumsum(self.noise_schedule, dim=0)[t])

    def _get_beta(self, t):
        """
        Compute beta value for the noise schedule.
        """
        return self.noise_schedule[t]

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
