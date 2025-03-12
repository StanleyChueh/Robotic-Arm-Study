# Robotic arm study
Imitation Learning with robosuite and robomimic

## Robomimic
#### Robot learning framework for training
![image](https://github.com/user-attachments/assets/4adadae3-5882-45c6-8557-365b44118f8b)

Install from Robomimic Official Repo
```
cd ~
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
python3.8 -m venv venv
source venv/bin/activate
pip install -e .
```

## Robosuite
#### Robot learning framework for data collection
![image](https://github.com/user-attachments/assets/d798a18e-443e-4d6b-a6ed-1734891b75e3)

Install from Robosuite Official Repo
```
cd ~
git clone https://github.com/ARISE-Initiative/robosuite.git -b 1.4.1
cd robosuite
pip3 install -r requirements.txt
```

## Imitation Learning Training Pipeline

### Collect Data
#### collect_human_demonstrations.py
```
cd ~/robosuite
export MUJOCO_GL=glfw
python robosuite/scripts/collect_human_demonstrations.py 
```

![image](https://github.com/user-attachments/assets/16909ae2-54bf-493f-8a80-1d5d574acde4)

### Replay Dataset

```
python robosuite/scripts/playback_demonstrations_from_hdf5.py --folder /path/to/demo.hdf5
```

### Convert to robomimic compatible format

After conversion,you can check the format to see what is changing.
```
cd ~/robomimic
source venv/bin/activate
python robomimic/scripts/conversion/convert_robosuite.py --dataset /path/to/demo.hdf5
```

Check dataset format
```
h5ls -r /path/to/demo.hdf5
```

### Extracting Observations from MuJoCo states

In this step, you will have image.hdf5 after extraction, and you are ready for the next step.

```
cd ~/robomimic
python robomimic/scripts/dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
```

You can also check dataset format

```
h5ls -r /path/to/demo.hdf5
```

### Training

Enable GPU while training
```
cd ~/robomimic
source venv/bin/activate
CUDA_VISIBLE_DEVICES=0  python robomimic/scripts/train.py --config /home/user/robomimic/robomimic/exps/templates/bc.json  --dataset /path/to/image.hdf5 
```

Tensorboard Monitoring

```
tensorboard --logdir=/path/to/logs/tb
```

### Inference

```
cd ~/robomimic
source venv/bin/activate
python robomimic/scripts/run_trained_agent.py     --agent /path/to/your.pth     --n_rollouts 5     --render
```
