# Robotic arm study
Study how to use robotic arm in ROS2 humble for imitation learning with robomimic

## Collect Data
![Untitled ‑ Made with FlexClip (65) (1)](https://github.com/user-attachments/assets/4d4a21df-dcb4-470e-a768-fdd9ec5a5d18)

## Train Imitation Learning model with Robomimic
![Untitled ‑ Made with FlexClip (62) (1)](https://github.com/user-attachments/assets/5a07200a-7833-497a-96d0-c1a13352654e)
epoch=50

![Untitled ‑ Made with FlexClip (63) (1)](https://github.com/user-attachments/assets/653bc5e4-406d-46b5-87b8-c10bab625d66)
epoch=100

![Untitled ‑ Made with FlexClip (64) (1)](https://github.com/user-attachments/assets/f3159607-75af-40f3-9e9b-a350b44df323)
epoch=150
 
## Usage

New Training Session 
``` 
source venv/bin/activate
 python robomimic/scripts/train.py --config robomimic/exps/templates/bc.json --dataset datasets/mujoco_lift_demo.hdf5     
```

Inference
```
python robomimic/scripts/run_trained_agent.py     --agent bc_trained_models/test/20250219090920/models/model_epoch_250.pth     --n_rollouts 5     --render
```
