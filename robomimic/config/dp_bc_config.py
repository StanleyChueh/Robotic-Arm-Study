"""
Config for DP_BC.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.models.diffusion_nets import DiffusionNetwork

class DP_BCConfig(BaseConfig):
    ALGO_NAME = "dp_bc"

    def experiment_config(self):
        """
        Update from subclass to set paper defaults for gym envs.
        """
        super(DP_BCConfig, self).experiment_config()

        # no validation and no video rendering
        self.experiment.validate = False
        self.experiment.render_video = False

        # save 10 checkpoints throughout training
        self.experiment.save.every_n_epochs = 20 

        # save models that achieve best rollout return instead of best success rate
        self.experiment.save.on_best_rollout_return = True
        self.experiment.save.on_best_rollout_success_rate = False

        # epoch definition - 5000 gradient steps per epoch, with 200 epochs = 1M gradient steps, and eval every 1 epochs
        self.experiment.epoch_every_n_steps = 5000

        # evaluate with normal environment rollouts
        self.experiment.rollout.enabled = True
        self.experiment.rollout.n = 50              # paper uses 10, but we can afford to do 50
        self.experiment.rollout.horizon = 1000
        self.experiment.rollout.rate = 1            # rollout every epoch to match paper

    def train_config(self):
        super(DP_BCConfig, self).train_config()
        print(f"🚀 DEBUG: Before setting, config.train.data = {self.train.data}")  

        # ✅ Ensure train.data is initialized
        if not hasattr(self.train, "data") or self.train.data is None:
            self.train.data = "datasets/mujoco_lift_demo.hdf5"  # Default dataset path

        print(f"✅ DEBUG: After setting, config.train.data = {self.train.data}")  

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """
        super(DP_BCConfig, self).algo_config()
        print("🚀 DEBUG: Assigning diffusion settings in algo_config")

        if not hasattr(self.algo, "diffusion"):
            self.algo.diffusion = {}

        self.algo.diffusion = {
            "enabled": True,
            "steps": 100,
            "noise_schedule": "cosine",
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "sampling_timesteps": 100,
            "predict_x0": True,
        }

        # Debugging: Verify diffusion is set correctly
        print(f"✅ DEBUG: self.algo.diffusion = {self.algo.diffusion}")

        # Optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # Learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # LR decay factor
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # LR decay schedule
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" # LR scheduler type
        self.algo.optim_params.policy.regularization.L2 = 0.00   

        # MLP network architecture (used inside the diffusion model)
        self.algo.actor_layer_dims = (1024, 1024)

        # RNN and Transformer settings (not used for DP_BC)
        self.algo.rnn.enabled = False
        self.algo.transformer.enabled = False

    def observation_config(self):
        """
        Update from superclass to use flat observations from gym envs.
        """
        super(DP_BCConfig, self).observation_config()

        # Enable joint positions, velocities, and end-effector pose
        self.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_joint_pos", "robot0_joint_vel"]
        
        # If using images, add RGB observations
        self.observation.modalities.obs.rgb = ["robot0_eye_in_hand_image"]  # ["agentview_image"] if using camera input
