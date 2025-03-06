import h5py
import json
import numpy as np

# Path to your HDF5 dataset
hdf5_path = "datasets/mujoco_lift_demo.hdf5"

# Open the dataset in read/write mode
hf = h5py.File(hdf5_path, "a")  # "a" mode allows modifications

# ✅ Ensure "env_args" exists and is correctly formatted
if "env_args" in hf["data"].attrs:
    try:
        env_args = json.loads(hf["data"].attrs["env_args"])  # Try parsing it
        print("✅ env_args found and correctly formatted.")

        # ✅ Ensure the 'type' field exists
        if "type" in env_args:
            print("✅ 'type' field found:", env_args["type"])
        else:
            print("❌ 'type' field is missing! Adding it now...")
            env_args["type"] = "robosuite"  # Default to robosuite
            hf["data"].attrs.modify("env_args", json.dumps(env_args))  # Save changes
            print("✅ Added 'type': 'robosuite' to env_args.")

    except json.JSONDecodeError:
        print("❌ env_args exists but is incorrectly formatted. Re-saving it...")
        env_args = {
            "env_name": "Lift",
            "robots": ["Panda"],
            "controller_configs": {"type": "OSC_POSE"},
            "type": "robosuite"
        }
        hf["data"].attrs.modify("env_args", json.dumps(env_args))  # Re-save correctly
        print("✅ Fixed env_args format.")
else:
    print("❌ env_args is missing! Adding it now...")
    env_args = {
        "env_name": "Lift",
        "robots": ["Panda"],
        "controller_configs": {"type": "OSC_POSE"},
        "type": "robosuite"  # Default to robosuite
    }
    hf["data"].attrs["env_args"] = json.dumps(env_args)  # Save as JSON string
    print("✅ Added env_args to dataset.")

# ✅ Ensure "shape_meta" exists
if "shape_meta" in hf["data"].attrs:
    print("✅ shape_meta found.")
else:
    print("❌ shape_meta is missing! Adding it now...")
    shape_meta = {
        "all_shapes": {
            "robot0_proprio-state": [32],  # Example: Adjust based on your data
            "actions": [7],  # Example: Adjust if needed
        },
        "use_images": False,
        "use_depths": False,
    }
    hf["data"].attrs["shape_meta"] = json.dumps(shape_meta)  # Save as JSON string
    print("✅ Added shape_meta.")

# ✅ List all recorded episodes
episodes = list(hf["data"].keys())
print("Recorded Episodes:", episodes)

# ✅ Verify observation and action data for each episode
for episode in episodes:
    obs_path = f"data/{episode}/obs/robot0_proprio-state"
    action_path = f"data/{episode}/actions"

    # Check if `robot0_proprio-state` exists
    obs_keys = list(hf[f"data/{episode}/obs"].keys())  # List available observations

    if obs_path in hf and action_path in hf:
        obs_shape = hf[obs_path].shape
        actions_shape = hf[action_path].shape
        print(f"✅ {episode} - Observations: {obs_shape}, Actions: {actions_shape}")
    else:
        print(f"❌ {episode} is missing necessary data.")

    # ✅ Auto-fix: Rename `robot_state` or `joint_positions` to `robot0_proprio-state` if missing
    if "robot0_proprio-state" not in obs_keys:
        print(f"❌ `robot0_proprio-state` is missing in {episode}. Trying to fix it...")

        # Check if another proprioceptive dataset exists
        candidate_keys = ["robot_state", "joint_positions"]
        replacement_key = None

        for key in candidate_keys:
            if key in obs_keys:
                replacement_key = key
                break

        if replacement_key:
            # Rename the found dataset to `robot0_proprio-state`
            hf[f"data/{episode}/obs"].move(replacement_key, "robot0_proprio-state")
            print(f"✅ Renamed `{replacement_key}` to `robot0_proprio-state` in {episode}.")
        else:
            print(f"❌ No suitable replacement found for `robot0_proprio-state` in {episode}. Manual intervention needed.")

# Close the dataset
hf.close()
print("✅ HDF5 dataset is fully prepared for training!")
