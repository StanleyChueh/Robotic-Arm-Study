
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch
import json
import time

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from robomimic.algo.diffusion_policy import DP_BC

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):

    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy) or isinstance(policy, DP_BC) 
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)
    print(f"🚀 DEBUG: Environment Action Space Range = {env.base_env.action_spec}")

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        env.render(mode="human")
        print(f"🚀 DEBUG: Environment Type = {type(env)}")
        print(f"🚀 DEBUG: Environment Attributes = {dir(env)}")  # List all available attributes


        for step_i in range(horizon):

            # get action from policy
            if isinstance(policy, DP_BC):
                obs_dict = {k: torch.tensor(v).unsqueeze(0).float().to(policy.device) for k, v in obs.items()}
                act = policy._sample_from_diffusion(obs_dict).squeeze(0).cpu().numpy()
            else:
                act = policy(ob=obs)  # Regular policy

            act = np.clip(act * 1.5, -1, 1)  # Increase force without going out of bounds

            # 🚨 Prevent invalid actions
            if np.isnan(act).any() or np.isinf(act).any():
                print("🚨 WARNING: NaN or Inf detected in action! Resetting to zero.")
                act = np.zeros_like(act)

            print(f"🚀 Step {step_i}: Action = {act}")
            print(f"🚀 Step {step_i}: Before Action - Joint Positions = {obs['robot0_joint_pos']}")
            print(f"🚀 Step {step_i}: Action Magnitude = {np.linalg.norm(act)}")

            next_obs, r, done, _ = env.step(act)

            print(f"🚀 Step {step_i}: After Action - Joint Positions = {next_obs['robot0_joint_pos']}")

            print(f"🚀 Step {step_i}: Next State = {next_obs}")

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            print(f"🚀 Step {step_i}: Reward = {r}, Done = {done}, Success = {success}")  

            # visualization
            if render:
                env.render(mode="human", camera_name="agentview")
                time.sleep(0.01)
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done:
                print(f"🚨 Rollout ended early at step {step_i}. Resetting environment.")
                break
            if success:
                print(f"✅ Success at step {step_i}!")
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
    print(f"✅ DEBUG: Final Stats - Total Reward = {total_reward}, Horizon = {step_i+1}, Success Rate = {success}")

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # 確保 ckpt_dict["config"] 轉換成字典
    # 確保 ckpt_dict["config"] 轉換為字典
    if "config" in ckpt_dict:
        if isinstance(ckpt_dict["config"], str):  # If config is a JSON string, convert it
            try:
                ckpt_dict["config"] = json.loads(ckpt_dict["config"])
            except json.JSONDecodeError as e:
                print(f"❌ ERROR: JSON parsing failed: {e}")
                exit(1)
        elif not isinstance(ckpt_dict["config"], dict):  # If it's not a dict, raise an error
            print(f"❌ ERROR: ckpt_dict['config'] is not a valid JSON dictionary: {type(ckpt_dict['config'])}")
            exit(1)
    else:
        print("❌ ERROR: ckpt_dict does not contain 'config'!")
        exit(1)

    config_dict = ckpt_dict["config"]

    if isinstance(policy.policy, DP_BC):
        print("🚀 DEBUG: Loaded Diffusion Policy Model!")
        diffusion_policy = policy.policy  # Extract underlying DP_BC model
        diffusion_policy.diffusion_steps = config_dict["algo"]["diffusion"]["steps"]
        # Extract noise_schedule
        noise_schedule = ckpt_dict["config"]["algo"]["diffusion"]["noise_schedule"]

        # 🚀 Debugging: Print the actual value and type
        print(f"🚀 DEBUG: noise_schedule -> {noise_schedule} (type: {type(noise_schedule)})")

        # Convert noise schedule if it's a string
        if isinstance(noise_schedule, str):
            if noise_schedule == "cosine":
                # Generate a cosine noise schedule
                timesteps = ckpt_dict["config"]["algo"]["diffusion"]["steps"]
                noise_schedule = np.cos(np.linspace(0, np.pi / 2, timesteps)) ** 2
                print(f"✅ DEBUG: Generated cosine noise schedule with shape {noise_schedule.shape}")
            else:
                raise ValueError(f"❌ ERROR: Unknown noise schedule type: {noise_schedule}")

        # Ensure noise_schedule is a valid list/array
        if not isinstance(noise_schedule, (list, np.ndarray)):
            raise TypeError(f"❌ ERROR: noise_schedule should be a list or numpy array, but got {type(noise_schedule)}")

        # Convert to PyTorch tensor
        diffusion_policy.noise_schedule = torch.tensor(noise_schedule, dtype=torch.float32).to(device)
        print(f"✅ Successfully set diffusion_policy.noise_schedule with shape {diffusion_policy.noise_schedule.shape}")

    else:
        print("❌ ERROR: Policy is not a DP_BC model. Check your checkpoint.")
        exit(1)  # Exit if the model is not a diffusion policy

    # Store as `config_dict`
    config_dict = ckpt_dict["config"]

    # Load diffusion steps
    diffusion_policy.diffusion_steps = config_dict["algo"]["diffusion"]["steps"]

    # Load noise schedule
    noise_schedule = config_dict["algo"]["diffusion"]["noise_schedule"]

    # Debugging
    print(f"🚀 DEBUG: noise_schedule -> {noise_schedule} (type: {type(noise_schedule)})")

    # Convert string noise schedule to an array
    if isinstance(noise_schedule, str) and noise_schedule == "cosine":
        timesteps = config_dict["algo"]["diffusion"]["steps"]
        noise_schedule = np.cos(np.linspace(0, np.pi / 2, timesteps)) ** 2
        print(f"✅ DEBUG: Generated cosine noise schedule with shape {noise_schedule.shape}")

    # Validate noise_schedule
    if not isinstance(noise_schedule, (list, np.ndarray)):
        raise TypeError(f"❌ ERROR: noise_schedule should be a list or numpy array, but got {type(noise_schedule)}")

    # Convert to PyTorch tensor
    diffusion_policy.noise_schedule = torch.tensor(noise_schedule, dtype=torch.float32).to(device)
    print(f"✅ Successfully set diffusion_policy.noise_schedule with shape {diffusion_policy.noise_schedule.shape}")

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        # Ensure config is properly assigned
        if isinstance(ckpt_dict["config"], dict):
            config = ckpt_dict["config"]  # Already a dictionary, use directly
        else:
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)  # Otherwise, load it properly

        rollout_horizon = config["experiment"]["rollout"]["horizon"]

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=True,  
        render_offscreen=False,  
        verbose=True,
    )
    # ✅ Ensure rendering is properly enabled
    env.viewer = None  # Force viewer initialization
    env.has_renderer = True  
    env.use_camera_obs = False  # Ensure no offscreen rendering interference

    # ✅ Ensure the environment supports rendering
    if not hasattr(env, "render"):
        print("❌ ERROR: Environment does not support rendering.")
        exit(1)


    if args.render:
        print("✅ DEBUG: Keeping render window open...")
        for i in range(args.n_rollouts):
            print(f"🚀 DEBUG: Running rollout {i+1} / {args.n_rollouts}")
            stats, traj = rollout(
                policy=policy, 
                env=env, 
                horizon=rollout_horizon, 
                render=args.render,  # ✅ Allow rendering during rollout
                video_writer=None, 
                video_skip=args.video_skip, 
                return_obs=False,
                camera_names=args.camera_names,
            )
            print(f"✅ DEBUG: Rollout {i+1} complete. Stats: {stats}")


    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    if not rollout_stats:
        print("❌ ERROR: No rollouts were collected. Check if `rollout()` is executing correctly.")
        exit(1)  # Stop execution

    # Ensure all rollout_stats entries have the same keys
    common_keys = set(rollout_stats[0].keys()) if rollout_stats else set()
    avg_rollout_stats = {k: np.mean([x[k] for x in rollout_stats if k in x]) for k in common_keys}

    # Compute total success count
    avg_rollout_stats["Num_Success"] = sum(x.get("Success_Rate", 0) for x in rollout_stats)

    print("✅ DEBUG: Final Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    run_trained_agent(args)
