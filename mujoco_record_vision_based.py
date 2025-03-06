import robosuite as suite
import numpy as np
import h5py
import pygame  # Install with: pip install pygame
import json

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

def get_human_action():
    """ Get human input from the keyboard and map it to a continuous action space. """
    action = np.zeros(env.action_dim)  # Initialize action array
    pygame.event.pump()  # Process keyboard events

    keys = pygame.key.get_pressed()

    move_speed = 0.2  # Adjust for smooth control
    grip_speed = 1.0  # Max speed for gripper

    # Adjusted movement mapping
    if keys[pygame.K_w]:  # Move forward
        action[0] = move_speed
    if keys[pygame.K_s]:  # Move backward
        action[0] = -move_speed
    if keys[pygame.K_a]:  # Move left
        action[1] = move_speed
    if keys[pygame.K_d]:  # Move right
        action[1] = -move_speed
    if keys[pygame.K_q]:  # Move down
        action[2] = -move_speed
    if keys[pygame.K_e]:  # Move up
        action[2] = move_speed
    
    # Use arrow keys for movement
    if keys[pygame.K_UP]:  # Move forward (+Y)
        action[2] = move_speed
    if keys[pygame.K_DOWN]:  # Move backward (-Y)
        action[2] = -move_speed


    # Open/close gripper
    if keys[pygame.K_o]:  # Open gripper
        action[-1] = -grip_speed
    if keys[pygame.K_p]:  # Close gripper
        action[-1] = grip_speed

    return action

num_episodes = 1
for ep in range(num_episodes):
    obs = env.reset()
    episode_group = demo_group.create_group(f"demo_{ep}")

    obs_list, action_list, reward_list, done_list = [], [], [], []

    print(f"Recording episode {ep+1}. Press ENTER to stop.")

    done = False
    while not done:
        env.render()
        action = get_human_action()  # Get user input
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

    # Save to HDF5 with all observation keys
    for key in obs_list[0].keys():
        data = np.array([o[key] for o in obs_list])

        if key == "robot0_eye_in_hand_image":  # ✅ Correct key for image storage
            data = data.astype(np.uint8)  # Convert to uint8
            print(f"Saving image data for {key} with shape: {data.shape}")
            episode_group.create_dataset(f"obs/{key}", data=data, compression="gzip")
        else:  # ✅ Save all other observation data normally
            episode_group.create_dataset(f"obs/{key}", data=data, compression="gzip")

    # Save dataset with compression to reduce HDF5 file size
    episode_group.create_dataset("actions", data=np.array(action_list))
    episode_group.create_dataset("rewards", data=np.array(reward_list))
    episode_group.create_dataset("dones", data=np.array(done_list))
    
    # ✅ Add num_samples attribute (required for robomimic)
    episode_group.attrs["num_samples"] = len(action_list)  # Store number of steps
    hf.flush()  # Ensure data is written before closing



print("All episodes recorded. Saving data...")
hf.close()
pygame.quit()
print("✅ Lift task demo recording complete!")
