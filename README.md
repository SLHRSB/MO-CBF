# MO-CBF: Multi-Output Control Barrier Function Safety Filter for Reinforcement Learning-Based Autonomous Driving

This repository contains the implementation of the **Multi-Output Control Barrier Function (MO-CBF)** framework proposed in:

> **[Paper Title]**
> Saeedeh Lohrasbi et al.
> *(will be updated with final citation after publication.)*

MO-CBF integrates a model-based safety filter with Deep Reinforcement Learning (DRL) to improve safety during both training and deployment. The framework combines a multi-objective Control Barrier Function (CBF) safety layer with an overriding penalty mechanism that encourages the RL agent to learn policies naturally aligned with safety constraints.

---

## Overview

Deep Reinforcement Learning agents can achieve high performance in autonomous driving tasks but often suffer from unsafe exploration and poor robustness under distribution shifts.

The proposed MO-CBF framework addresses these challenges by:

* Enforcing safety constraints through a runtime CBF safety filter
* Simultaneously regulating steering, throttle, and braking actions
* Penalizing safety interventions during training through an overriding penalty
* Preserving safety under changing vehicle dynamics and cyberattack scenarios
* Improving policy robustness and generalization

The framework was evaluated in high-fidelity autonomous driving simulations and compared against standard PPO-based baselines.

---

## Key Features

### Safe Reinforcement Learning

* PPO-based driving policy
* Runtime safety enforcement using Control Barrier Functions
* Multi-objective action correction

### Overriding Penalty

During training, a penalty is applied whenever the CBF modifies the RL-generated action:

```math
R_P = -\xi \left\| a_{RL} - a_{CBF} \right\|_1
```

where:

* `a_RL` is the action proposed by the reinforcement learning policy
* `a_CBF` is the action after safety filtering
* `ξ` is the penalty coefficient

This mechanism encourages the policy to internalize safe behavior and reduce future interventions.

### Robustness Evaluation

The framework is evaluated under multiple testing scenarios:

* Baseline driving
* Physics parameter changes
* Denial-of-Service (DoS) attacks
* Sensor perturbations
* Safety ablation studies

---

## Repository Structure

```text
MO-CBF/
│
├── configs/           # Experiment configurations
├── src/
│   ├── agents/        # PPO agents
│   ├── cbf/           # MO-CBF implementation
│   ├── envs/          # Driving environments
│   ├── attacks/       # Adversarial attack modules
│   └── utils/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── plot_results.py
│
├── results/
├── figures/
├── docs/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/SLHRSB/MO-CBF.git
cd MO-CBF
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
conda activate mocbf
```

---

## Training

Example:

```bash
python scripts/train.py --config configs/B2.yaml
```

Available agents:

| Agent | Description                      |
| ----- | -------------------------------- |
| A1    | PPO baseline                     |
| A2    | PPO with transfer learning       |
| B1    | PPO + MO-CBF                     |
| B2    | PPO + MO-CBF + transfer learning |

---

## Evaluation

Run evaluation:

```bash
python scripts/evaluate.py --config configs/B2.yaml
```

Reported metrics include:

* Episode reward
* Collision rate
* Average speed
* Safety interventions
* CBF correction statistics

---

## Reproducing Paper Results

| Result                       | Script                       |
| ---------------------------- | ---------------------------- |
| Training curves              | `plot_training_results.py`   |
| Collision rate analysis      | `evaluate_collision_rate.py` |
| Safety intervention analysis | `plot_cbf_interventions.py`  |
| Robustness experiments       | `evaluate_robustness.py`     |

---

## Safety Intervention Analysis

The repository reports both:

1. Cumulative intervention counts
2. Significant action correction rates

The second metric provides a more meaningful measure of policy safety alignment by quantifying how frequently the CBF performs substantial modifications to the RL-generated action.

---

## Limitations

Current implementation has several limitations:

* Explicit fixed and stochastic sensor-delay experiments are not included.
* Results may vary across simulators and hardware configurations.
* Hyperparameters were tuned for the scenarios reported in the paper.
* Additional evaluation under broader traffic conditions remains future work.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{,
  title={Multi-Output Control Barrier Function Safety Filter for Reinforcement Learning-Based Autonomous Driving},
  author={Lohrasbi, Saeedeh and others},
  journal={TBD},
  year={2026}
}
```

---

## Contact

**Saeedeh Lohrasbi**
PhD Candidate, Systems Design Engineering
University of Waterloo

For questions, bug reports, or collaboration opportunities, please open an issue or contact the authors.
