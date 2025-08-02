from typing import Tuple, Optional, List, Dict, Any

import numpy as np

from nesenv.envs.NESEnvironment import NESEnvironment


class PacManEnvironment(NESEnvironment):
    """
    Pac-Man specific NES environment that uses score from RAM as reward.

    This environment reads the score directly from the game's RAM memory
    and uses score differences as the primary reward signal.
    """

    LIVES_ADDRESS = 0x0067
    LEVEL_ADDRESS = 0x0068
    SCORE_ADDRESS_1 = 0x0070
    SCORE_ADDRESS_2 = 0x0071
    SCORE_ADDRESS_3 = 0x0072
    SCORE_ADDRESS_4 = 0x0073
    SCORE_ADDRESS_5 = 0x0074
    SCORE_ADDRESS_6 = 0x0075

    def __init__(
            self,
            rom_path: str,
            frame_skip: int = 4,
            frame_stack: int = 4,
            time_between_frames: float = 0.1,
            grayscale: bool = True,
            resize_shape: Tuple[int, int] = (84, 84),
            max_episode_steps: int = 10000,
            score_reward_scale: float = 0.01,
            life_penalty: float = -100.0,
            level_bonus: float = 1000.0,
            action_map: Optional[List[List[int]]] = None,
    ):
        """
        Initialize the Pac-Man environment.

        Args:
            rom_path: Path to the Pac-Man NES ROM file
            frame_skip: Number of frames to skip between actions
            frame_stack: Number of frames to stack in observation
            time_between_frames: Time between getting frame data
            grayscale: Convert frames to grayscale
            resize_shape: Resize frames to this shape (height, width)
            max_episode_steps: Maximum steps per episode
            score_reward_scale: Scale factor for score-based rewards
            life_penalty: Penalty when losing a life
            level_bonus: Bonus when completing a level
            action_map: Custom action mapping for discrete actions
        """

        if action_map is None:
            action_map = [
                [],
                [self.UP],
                [self.DOWN],
                [self.LEFT],
                [self.RIGHT],
            ]

        super().__init__(
            rom_path=rom_path,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            time_between_frames=time_between_frames,
            grayscale=grayscale,
            resize_shape=resize_shape,
            max_episode_steps=max_episode_steps,
            reward_function=self._pacman_reward_function,
            action_map=action_map,
        )

        self.score_reward_scale = score_reward_scale
        self.life_penalty = life_penalty
        self.level_bonus = level_bonus

        self.previous_score = 0
        self.previous_lives = 3
        self.previous_level = 1
        self.game_over = False

    def _read_ram_address(self, address: int) -> int:
        """
        Read a value from RAM at the specified address.

        Args:
            address: Memory address to read from

        Returns:
            Value at the memory address (0-255)
        """
        try:
            return self.client.get_value_at_address(address)
        except Exception as e:
            print(f"Error reading RAM address 0x{address:04X}: {e}")
            return 0

    def _get_score(self) -> int:
        """
        Read the current score from RAM.
        """
        try:
            digit_1 = self._read_ram_address(self.SCORE_ADDRESS_1) & 0x0F
            digit_2 = self._read_ram_address(self.SCORE_ADDRESS_2) & 0x0F
            digit_3 = self._read_ram_address(self.SCORE_ADDRESS_3) & 0x0F
            digit_4 = self._read_ram_address(self.SCORE_ADDRESS_4) & 0x0F
            digit_5 = self._read_ram_address(self.SCORE_ADDRESS_5) & 0x0F
            digit_6 = self._read_ram_address(self.SCORE_ADDRESS_6) & 0x0F

            score = (digit_6 * 1000000 +
                     digit_5 * 100000 +
                     digit_4 * 10000 +
                     digit_3 * 1000 +
                     digit_2 * 100 +
                     digit_1 * 10)

            return score
        except Exception as e:
            print(f"Error reading score: {e}")
            return self.previous_score

    def _get_lives(self) -> int:
        """Read the current number of lives from RAM."""
        return self._read_ram_address(self.LIVES_ADDRESS)

    def _get_level(self) -> int:
        """Read the current level from RAM."""
        return self._read_ram_address(self.LEVEL_ADDRESS)

    def _is_game_over(self) -> bool:
        """Check if the game is over (no lives remaining)."""
        return self._get_lives() == 0

    def _pacman_reward_function(self, _: np.ndarray, __: np.ndarray) -> float:
        """
        Pac-Man specific reward function based on score changes and game events.

        Rewards:
        - Score increase reward (eating dots, power pellets, ghosts, fruits)
        - Life loss penalty
        - Level bonus
        """

        current_score = self._get_score()
        current_lives = self._get_lives()
        current_level = self._get_level()

        reward = 0.0

        score_diff = current_score - self.previous_score
        if score_diff > 0:
            reward += score_diff * self.score_reward_scale
            print(f"Score increased by {score_diff}! Total score: {current_score}")

        if current_lives < self.previous_lives:
            reward += self.life_penalty
            print(f"Life lost! Lives remaining: {current_lives}, Penalty: {self.life_penalty}")

        if current_level != 255 and current_level > self.previous_level:
            reward += self.level_bonus
            print(f"New level started! Level: {current_level}, Bonus: {self.level_bonus}")

        self.previous_score = current_score
        self.previous_lives = current_lives
        self.previous_level = current_level

        return reward

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and initialize game state tracking."""
        observation, info = super().reset(seed=seed, options=options)

        self.previous_score = self._get_score()
        self.previous_lives = self._get_lives()
        self.previous_level = self._get_level()
        self.game_over = False

        info.update({
            "score": self.previous_score,
            "lives": self.previous_lives,
            "level": self.previous_level,
            "game_over": self.game_over,
        })

        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with Pac-Man specific termination conditions."""
        observation, reward, terminated, truncated, info = super().step(action)

        self.game_over = self._is_game_over()
        terminated = terminated or self.game_over

        current_score = self._get_score()
        current_lives = self._get_lives()
        current_level = self._get_level()

        info.update({
            "score": current_score,
            "lives": current_lives,
            "level": current_level,
            "game_over": self.game_over,
            "score_reward": (current_score - self.previous_score) * self.score_reward_scale,
        })

        return observation, reward, terminated, truncated, info

    def get_game_state(self) -> Dict[str, Any]:
        """Get detailed game state information."""
        return {
            "score": self._get_score(),
            "lives": self._get_lives(),
            "level": self._get_level(),
            "game_over": self._is_game_over(),
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
        }