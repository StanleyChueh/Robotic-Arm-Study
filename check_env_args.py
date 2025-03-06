import h5py
import json

# Open your dataset
hdf5_path = "datasets/mujoco_lift_demo.hdf5"
hf = h5py.File(hdf5_path, "r")

# Check if 'env_args' exists
if "env_args" in hf["data"].attrs:
    env_args = json.loads(hf["data"].attrs["env_args"])  # Load JSON
    print("✅ env_args found:", env_args)

    if "type" in env_args:
        print("✅ 'type' field found:", env_args["type"])
    else:
        print("❌ 'type' field is missing!")

    if "env_kwargs" in env_args:
        print("✅ 'env_kwargs' field found:", env_args["env_kwargs"])
    else:
        print("❌ 'env_kwargs' field is missing!")

else:
    print("❌ env_args is missing!")

hf.close()
