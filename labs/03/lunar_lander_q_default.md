# Model pedigree

Pedigree of the model `lunar_lander_q_default.py`

## Model 0

* Cluster job id: 345245
* Git commit: 2258cebafaaeb805ab8ffdf8ac7aac9d4c22e3af
* Call: `lunar_lander.py --episodes 100000 --expert_trajectories 100000 --format npy --output slurm-345245-model.npy`
* The model was only trained on the expert trajectories (and not the interactive play).

## Model 1

* Cluster job id: 345350
* Git commit: c18694a4ba0a7b02ae8d685074766cac10b5487a
* Call: `lunar_lander.py --episodes 10000 -i lunar_lander_q_trajectories.npy --train --alpha 0.1 --output slurm-345350-model.npy`
* Score: 127.64

## Model 2

* Cluster job id: 345488
* Git commit: 24fe39e6cbd74a236c3a5c0a2b59c6aa0510fd8b
* Call: `lunar_lander.py --episodes 10000 --train --alpha 0.1 --output slurm-345488-model.npy --run_name slurm-345488-lunar_lander_cont_10k_a01`
* Score: 216.53

## Model 3

* Cluster job id: 345500
* Git commit: 24fe39e6cbd74a236c3a5c0a2b59c6aa0510fd8b
* Call: `lunar_lander.py --episodes 10000 --train --alpha 0.05 -i slurm-345488-model.npy --output slurm-345500-model.npy --run_name slurm-345500-lunar_lander_cont2_10k_a005`
* Score: 229.49
