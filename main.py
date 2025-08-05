import gymnasium
import nesenv

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

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

def train_pacman_environment():
    """Train agent in the Pac-Man environment."""

    env = gymnasium.make(
        'nesenv/PacManEnv-v0',
        rom_path="/home/stefan/Dev/nesrs/assets/pacman-level1.cpu",
        frame_skip=50000,
        frame_stack=2,
        score_reward_scale=0.5,
        life_penalty=-20.0,
        level_bonus=1000.0,
        max_episode_steps=10000,
        resize_shape=(128,120)
    )

    model = DQN("CnnPolicy", env, verbose=1, exploration_fraction=0.2,tensorboard_log="./pacman_tensorboard",buffer_size=50000,learning_rate=5e-4)

    model.learn(
        total_timesteps=1000000,
        log_interval=4,
        progress_bar=True,
    )

    model.save("pacman_model")
    print("Testing trained model...")
    test_model(model, env)

def test_model(model, env, episodes=5):
    """Test the trained model for several episodes."""
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}:")
        print(f"Initial - Score: {info.get('score', 0)}, Lives: {info.get('lives', 3)}")

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if steps % 100 == 0:
                print(f"Step {steps} - Score: {info.get('score', 0)}, "
                      f"Lives: {info.get('lives', 0)}, Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"Episode ended - Total steps: {steps}, "
                      f"Final score: {info.get('score', 0)}, "
                      f"Total reward: {total_reward:.2f}")
                break

if __name__ == "__main__":
    train_pacman_environment()