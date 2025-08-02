# NES Environment for Reinforcement Learning

## Overview
This project provides a Gymnasium environment for NES games, with current example tailored for Pac-Man. The environment connects to a Rust-based NES emulator backend and provides:
- Frame-based observations
- Customizable action spaces
- RAM memory access for game state tracking

## Features
- Multiple observation modes:
  - Raw RGB frames or grayscale
  - Frame stacking
  - Resizable observations
  - Configurable frame skipping
- Reinforcement learning ready (compatible with Stable Baselines3):

## Requirements
- Python 3.8+
- gymnasium
- numpy
- opencv-python
- stable-baselines3 (or some other RL library)
- nesrs (Rust emulator backend)

## Installation
1. Ensure you have Python 3.8+ installed
2. Clone the repository:
```bash
git clone https://github.com/StefanJo3107/nesenv
cd nesenv
```
3. Install packages:
```bash
pip install -e .
```
4. Install package for emulator bindings through maurin (more info [here](https://github.com/StefanJo3107/nesrs/tree/master?tab=readme-ov-file#python-bindings))

## Usage
### Basic environment testing
```python
    import gymnasium
    import nesenv

    env = gymnasium.make(
        'nesenv/PacManEnv-v0',
        rom_path="path_to_rom_or_save_state",
        frame_skip=100000,
        frame_stack=4,
        score_reward_scale=0.01,
        life_penalty=-100.0,
        level_bonus=1000.0,
        max_episode_steps=30000
    )

    observation, info = env.reset()
    print(f"Initial state: {info}")

    total_reward = 0
    step_count = 0

    while True:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        if step_count % 100 == 0:
            print(f"Step {step_count}: Score={info['score']}, Lives={info['lives']}, "
                  f"Level={info['level']}, "
                  f"Reward={reward:.2f}")

        if terminated or truncated:
            print(f"Episode finished after {step_count} steps")
            print(f"Final score: {info['score']}")
            print(f"Total reward: {total_reward:.2f}")
            break

    env.close()
```

### Training and evaluating an Agent
```python
    import gymnasium
    import nesenv
    from stable_baselines3 import DQN

    env = gymnasium.make(
        'nesenv/PacManEnv-v0',
        rom_path="path_to_rom_or_save_state",
        frame_skip=100000,
        frame_stack=4,
        score_reward_scale=0.01,
        life_penalty=-100.0,
        level_bonus=1000.0,
        max_episode_steps=30000,
        resize_shape=(40,40)
    )

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("pacman_model")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
```

## Environment configuration parameters

| Parameter               | Type          | Default      | Description |
|-------------------------|---------------|--------------|-------------|
| `rom_path`              | str           | -            | Path to NES ROM file (required) |
| `frame_skip`            | int           | 4            | Number of frames to skip between actions |
| `frame_stack`           | int           | 4            | Number of frames to stack in observation |
| `time_between_frames`   | float         | 0.1          | Minimum time between frame updates (seconds) |
| `grayscale`             | bool          | True         | Convert frames to grayscale |
| `resize_shape`          | Tuple[int,int]| (84, 84)     | Target size for resizing frames (height, width) |
| `max_episode_steps`     | int           | 10000        | Maximum steps per episode |
| `score_reward_scale`    | float         | 0.01         | Multiplier for score-based rewards (Pac-Man only) |
| `life_penalty`          | float         | -100.0       | Reward penalty when losing a life (Pac-Man only) |
| `level_bonus`           | float         | 1000.0       | Reward bonus when completing a level (Pac-Man only) |
| `action_map`            | List[List[int]]| None        | Custom action mapping (buttons combinations for each action) |

## Extending for Other Games
To create an environment for another NES game:
- Create a new class inheriting from NESEnvironment
- Implement game-specific RAM address constants
- Override the reward function
- Add any game-specific state tracking
