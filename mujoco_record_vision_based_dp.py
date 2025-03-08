import robosuite as suite
import numpy as np
import h5py
import pygame  # Install with: pip install pygame
import json
from scipy.spatial.transform import Rotation as R

# Initialize pygame for keyboard input
pygame.init()
screen = pygame.display.set_mode((300, 300))  # Required to make pygame detect events

# Load the Lift task with proper rendering
env = suite.make(
    env_name="Lift",
    robots="Panda",
    use_camera_obs=True,  # Disable images for low-dim training
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=True,  # Enables on-screen rendering
    render_camera="robot0_eye_in_hand",  # Specify a camera for rendering
    camera_names=["robot0_eye_in_hand"],  # Cure images from the "birdview" camera
    camera_heights=84,  # Match the config image size
    camera_widths=84,
    camera_depths=False  # Set to True if you also want depth images
)

# Create an HDF5 file to save the demonstration
hdf5_path = "datasets/mujoco_lift_demo.hdf5"
hf = h5py.File(hdf5_path, "w")
demo_group = hf.create_group("data")

# ✅ Add required metadata for Robomimic compatibility
env_args = {
    "env_name": "Lift",
    "env_version": "1.4.1",
    "type": 1,  # Matches official dataset (integer)
    "env_kwargs": {
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "camera_names": ["robot0_eye_in_hand"],
        "control_freq": 20,
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2
        },
        "robots": ["Panda"],
        "camera_depths": False,
        "camera_heights": 84,
        "camera_widths": 84,
        "reward_shaping": False
    }
}
hf["data"].attrs["env_args"] = json.dumps(env_args)

# ✅ Define shape metadata dynamically
shape_meta = {
    "all_shapes": {
        "robot0_proprio-state": [env.observation_spec()["robot0_proprio-state"].shape[0]],
        "robot0_eef_pos": [3],
        "robot0_eef_quat": [4],
        "robot0_eef_vel_ang": [3],
        "robot0_eef_vel_lin": [3],
        "robot0_gripper_qpos": [2],
        "robot0_gripper_qvel": [2],
        "robot0_joint_pos": [7],
        "robot0_joint_pos_cos": [7],
        "robot0_joint_pos_sin": [7],
        "robot0_joint_vel": [7],
        "object": [10],
        "actions": [env.action_dim],  
        "rgb": [84, 84, 3]  # ✅ Include image shape (H, W, Channels)
    },
    "use_images": True,  # ✅ Enable image use
    "use_depths": False,
}
hf["data"].attrs["shape_meta"] = json.dumps(shape_meta)
hf.flush()

print("✅ Metadata added: env_args and shape_meta.")

# ✅ Function to get multiple observations
def get_observations(obs):
    """ Extract both state and image observations. """
    print("Available observation keys:", obs.keys())
    return {
        "robot0_proprio-state": obs["robot0_proprio-state"],
        "robot0_eef_pos": obs.get("robot0_eef_pos", np.zeros(3)),
        "robot0_eef_quat": obs.get("robot0_eef_quat", np.zeros(4)),
        "robot0_eef_vel_ang": obs.get("robot0_eef_vel_ang", np.zeros(3)),
        "robot0_eef_vel_lin": obs.get("robot0_eef_vel_lin", np.zeros(3)),
        "robot0_gripper_qpos": obs.get("robot0_gripper_qpos", np.zeros(2)),
        "robot0_gripper_qvel": obs.get("robot0_gripper_qvel", np.zeros(2)),
        "robot0_joint_pos": obs.get("robot0_joint_pos", np.zeros(7)),
        "robot0_joint_pos_cos": np.cos(obs.get("robot0_joint_pos", np.zeros(7))),
        "robot0_joint_pos_sin": np.sin(obs.get("robot0_joint_pos", np.zeros(7))),
        "robot0_joint_vel": obs.get("robot0_joint_vel", np.zeros(7)),
        "object": obs.get("object-state", np.zeros(10)), 
        "robot0_eye_in_hand_image": obs.pop("robot0_eye_in_hand_image", None)  # If using birdview camera
    }

def get_human_action(obs):
    action = np.zeros(env.action_dim)
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    move_speed = 0.2
    grip_speed = 1.0
    rotation_speed = 0.3

    # Step 1: Get current orientation (clearly converting quaternion from robosuite to scipy)
    gripper_quat_robosuite = obs["robot0_eef_quat"]
    gripper_quat_scipy = np.array([
        gripper_quat_robosuite[1],  # x
        gripper_quat_robosuite[2],  # y
        gripper_quat_robosuite[3],  # z
        gripper_quat_robosuite[0]   # w
    ])

    # Rotation from gripper frame to world frame
    gripper_rot_matrix = R.from_quat(gripper_quat_scipy).as_matrix()

    # Step 1: Define movement clearly in camera/gripper local frame
    move_local_frame = np.zeros(3)
    move_speed = 0.2
    grip_speed = 1.0
    rotation_speed = 0.3

    pygame.event.pump()
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:  # Forward (local X-axis of camera)
        move_local_frame[0] += move_speed
    if keys[pygame.K_s]:  # Backward
        move_local_frame[0] -= move_speed
    if keys[pygame.K_a]:  # Left (local Y-axis)
        move_local_frame[1] += move_speed
    if keys[pygame.K_d]:  # Right
        move_local_frame[1] -= move_speed
    if keys[pygame.K_UP]:  # Up (local Z-axis)
        move_local_frame[2] += move_speed
    if keys[pygame.K_DOWN]:  # Down
        move_local_frame[2] -= move_speed

    # Step 2: Rotate local movements clearly into global frame
    # ✅ Clearly critical step to maintain consistent intuitive controls
    move_global_frame = gripper_rot_matrix @ move_local_frame

    # Assemble final action clearly
    action = np.zeros(env.action_dim)
    action[:3] = move_global_frame

    # Wrist rotation (yaw clearly around local Z-axis)
    if keys[pygame.K_r]:
        action[5] += rotation_speed

    # Gripper open/close
    if keys[pygame.K_o]:
        action[-1] = -grip_speed
    if keys[pygame.K_p]:
        action[-1] = grip_speed

    return action

num_episodes = 5
for ep in range(num_episodes):
    obs = env.reset()
    episode_group = demo_group.create_group(f"demo_{ep}")

    obs_list, action_list, reward_list, done_list = [], [], [], []

    print(f"Recording episode {ep+1}. Press ENTER to stop.")

    done = False
    while not done:
        env.render()
        action = get_human_action(obs)  # Get user input
        next_obs, reward, done, info = env.step(action)

        # ✅ Store multiple observations
        obs_list.append(get_observations(obs))
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(done)

        obs = next_obs  # Move to next step

        # ✅ Stop recording manually
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:  # Press ENTER to stop episode
            print(f"Manually stopping episode {ep+1}.")
            done = True

    Tp = 100  # prediction horizon for diffusion policy

    # At the end of each episode:
    # ✅ Fix shape alignment for HDF5 format
    obs_seqs = {key: np.array([o.get(key, np.zeros_like(o[key])) for o in obs_list[:-Tp]]) for key in obs_list[0].keys()}
    next_obs_seqs = {key: np.array([o.get(key, np.zeros_like(o[key])) for o in obs_list[1:-Tp+1]]) for key in obs_list[0].keys()}
    action_seqs = np.array(action_list[:-Tp], dtype=np.float32)  # Ensure shape [T, action_dim]
    reward_seqs = np.array(reward_list[:-Tp], dtype=np.float32)
    done_seqs = np.array(done_list[:-Tp], dtype=np.float32)

    # ✅ Store observations properly
    for key in obs_list[0].keys():
        episode_group.create_dataset(f"obs/{key}", data=obs_seqs[key], compression="gzip")
        episode_group.create_dataset(f"next_obs/{key}", data=next_obs_seqs[key], compression="gzip")

    # ✅ Store actions correctly
    episode_group.create_dataset("actions", data=action_seqs, compression="gzip")

    # ✅ Store rewards and dones properly
    episode_group.create_dataset("rewards", data=reward_seqs, compression="gzip")
    episode_group.create_dataset("dones", data=done_seqs, compression="gzip")

    # ✅ Fix num_samples attribute
    episode_group.attrs["num_samples"] = len(obs_seqs[list(obs_seqs.keys())[0]])

    hf.flush()  # Ensure data is written before closing


print("All episodes recorded. Saving data...")
hf.close()
pygame.quit()
print("✅ Lift task demo recording complete!")
