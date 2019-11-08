# `lunar_lander_q_default.py`

* Cluster job id: 345350
* Git commit: c18694a4ba0a7b02ae8d685074766cac10b5487a
* Call: `lunar_lander.py --episodes 10000 -i lunar_lander_q_trajectories.npy --train --alpha 0.1 --output slurm-345350-model.npy`
* Score: 127.64

## `lunar_lander_q_trajectories.npy`

* Cluster job id: 345245
* Git commit: 2258cebafaaeb805ab8ffdf8ac7aac9d4c22e3af
* Call: `lunar_lander.py --episodes 100000 --expert_trajectories 100000 --format npy --output slurm-345245-model.npy`
* The model was only trained on the expert trajectories (and not the interactive play).
