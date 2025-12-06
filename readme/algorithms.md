# Algorithms

`tmrl` supports the following algorithms:
- **SAC** (Soft Actor-Critic)
- **REDQ-SAC** (Randomized Ensembled Double Q-Learning)
- **TD3** (Twin Delayed DDPG)

## Configuration

To select an algorithm, set the `ALGORITHM` key in the `ALG` section of your `config.json`.

### TD3 Configuration Example

```json
"ALG": {
    "ALGORITHM": "TD3",
    "LR_ACTOR": 0.00001,
    "LR_CRITIC": 0.00005,
    "GAMMA": 0.995,
    "POLYAK": 0.995,
    "ACTION_NOISE": 0.1,
    "TARGET_NOISE": 0.2,
    "NOISE_CLIP": 0.5,
    "POLICY_DELAY": 2,
    "OPTIMIZER_ACTOR": "adam",
    "OPTIMIZER_CRITIC": "adam"
},
```
