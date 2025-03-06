import h5py
import json

with h5py.File("/home/stanley/robomimic/datasets/lift/ph/demo.hdf5", "r") as f:
    print("Metadata:", json.loads(f["data"].attrs["env_args"]))
    print("Shape Meta:", json.loads(f.attrs["shape_meta"]))
    print("Observation keys stored:", list(f["data/demo_0/obs"].keys()))
    print("Sample image shape:", f["data/demo_0/obs/rgb"].shape)
