import time

import gymnasium
import nesenv

def test_pacman_environment():
    """Test the Pac-Man environment with score-based rewards."""

    env = gymnasium.make(
        'nesenv/PacManEnv-v0',
        rom_path="/home/stefan/Dev/nesrs/assets/pacman-level1.cpu",
        frame_skip=100000,
        frame_stack=4,
        score_reward_scale=0.01,
        life_penalty=-100.0,
        level_bonus=1000.0,
        max_episode_steps=30000
    )

    time.sleep(1)
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

if __name__ == "__main__":
    test_pacman_environment()