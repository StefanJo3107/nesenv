import time
from typing import Tuple, Optional, List, Dict, Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nesenv.emulator.NESEmulatorClient import NESEmulatorClient


class NESEnvironment(gym.Env):
    """
    Gymnasium environment for NES games using the Rust emulator backend.

    This environment provides a standard RL interface for NES games with
    configurable action spaces, observation preprocessing, and reward functions.
    """

    UP = 0b00010000
    DOWN = 0b00100000
    LEFT = 0b01000000
    RIGHT = 0b10000000
    BUTTON_A = 0b00000001
    BUTTON_B = 0b00000010
    SELECT = 0b00000100
    START = 0b00001000

    BUTTONS = [UP, DOWN, LEFT, RIGHT, SELECT, START, BUTTON_A, BUTTON_B]

    def __init__(
        self,
        rom_path: str,
        frame_skip: int = 4,
        frame_stack: int = 4,
        time_between_frames: float = 0.1,
        grayscale: bool = True,
        resize_shape: Tuple[int, int] = (84, 84),
        max_episode_steps: int = 10000,
        reward_function: Optional[callable] = None,
        action_map: Optional[List[List[int]]] = None,
    ):
        """
        Initialize the NES environment.

        Args:
            rom_path: Path to the NES ROM file
            frame_skip: Number of frames to skip between actions
            frame_stack: Number of frames to stack in observation
            time_between_frames: Time between getting frame data
            grayscale: Convert frames to grayscale
            resize_shape: Resize frames to this shape (height, width)
            max_episode_steps: Maximum steps per episode
            reward_function: Custom reward function (frame, prev_frame) -> reward
            action_map: Custom action mapping for discrete actions
        """
        super().__init__()

        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.time_between_frames = time_between_frames
        self.grayscale = grayscale
        self.resize_shape = resize_shape
        self.max_episode_steps = max_episode_steps
        self.reward_function = reward_function or self._default_reward_function

        self.client = None
        self._connect_client()

        self._setup_action_space(action_map)

        self._setup_observation_space()

        self.episode_step = 0
        self.total_reward = 0.0
        self.last_frame = None
        self.last_frame_time = time.time()
        self.frame_buffer = []
        self.last_action = None

    def _connect_client(self):
        """Connect to the NES emulator server."""
        try:
            self.client = NESEmulatorClient()
            self.client.load_rom(self.rom_path)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to NES emulator server: {e}")

    def _setup_action_space(self, action_map: Optional[List[List[int]]]):
        """Setup the action space"""
        if action_map is None:
            self.action_map = [
                [],
                [self.UP],
                [self.DOWN],
                [self.LEFT],
                [self.RIGHT],
                [self.BUTTON_A],
                [self.BUTTON_B],
                [self.BUTTON_A, self.UP],
                [self.BUTTON_A, self.DOWN],
                [self.BUTTON_A, self.LEFT],
                [self.BUTTON_A, self.RIGHT],
                [self.BUTTON_B, self.UP],
                [self.BUTTON_B, self.DOWN],
                [self.BUTTON_B, self.LEFT],
                [self.BUTTON_B, self.RIGHT],
            ]
        else:
            self.action_map = action_map
            self.action_space = spaces.Discrete(len(self.action_map))

    def _setup_observation_space(self):
        """Setup the observation space"""
        if self.grayscale:
            channels = 1
        else:
            channels = 3

        channels *= self.frame_stack

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.resize_shape[0], self.resize_shape[1], channels),
            dtype=np.uint8
        )

    def _preprocess_frame(self, frame: Optional[np.ndarray]) -> np.ndarray:
        """Preprocess a single frame."""
        if frame is None:
            if self.grayscale:
                return np.zeros((*self.resize_shape, 1), dtype=np.uint8)
            else:
                return np.zeros((*self.resize_shape, 3), dtype=np.uint8)

        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if frame.shape[:2] != self.resize_shape:
            frame = cv2.resize(frame, (self.resize_shape[1], self.resize_shape[0]))

        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)

        return frame.astype(np.uint8)

    def _get_observation(self) -> np.ndarray:
        """Get the current observation (stacked frames)."""
        if len(self.frame_buffer) == 0:
            for _ in range(self.frame_stack):
                black_frame = self._preprocess_frame(None)
                self.frame_buffer.append(black_frame)

        observation = np.concatenate(self.frame_buffer, axis=-1)
        return observation

    def _apply_action(self, action):
        """Apply an action to the environment."""
        for button in self.BUTTONS:
            self.client.release_key(button)

        buttons_to_press = self.action_map[action]
        for button in buttons_to_press:
            self.client.press_key(button)

    def _default_reward_function(self, frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """
        Default reward function - override this for specific games.

        This implementation gives a small positive reward for any frame change
        to encourage exploration.
        """
        if prev_frame is None:
            return 0.0

        if frame is not None and prev_frame is not None:
            diff = np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)))
            return diff / 255.0 * 0.01

        return 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        self.client.reset()

        self.episode_step = 0
        self.total_reward = 0.0
        self.last_frame = None
        self.last_frame_time = time.time()
        self.frame_buffer = []

        initial_frame = self.client.get_frame()
        processed_frame = self._preprocess_frame(initial_frame)

        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame.copy())

        self.last_frame = initial_frame

        observation = self._get_observation()
        info = {
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
        }

        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.episode_step += 1

        self._apply_action(action)

        frames = []
        for _ in range(self.frame_skip):
            self.client.step()
            if time.time() - self.last_frame_time > self.time_between_frames:
                frame = self.client.get_frame()
                if frame is not None:
                    frames.append(frame)
                self.last_frame_time = time.time()

        current_frame = frames[-1] if frames else None

        reward = self.reward_function(current_frame, self.last_frame)
        self.total_reward += reward

        processed_frame = self._preprocess_frame(current_frame)
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)

        observation = self._get_observation()

        terminated = False
        truncated = self.episode_step >= self.max_episode_steps

        self.last_frame = current_frame

        info = {
            "episode_step": self.episode_step,
            "total_reward": self.total_reward,
            "frame_skip": self.frame_skip,
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the environment."""
        pass
