import mmap
import struct
import time
from typing import Dict, Any, Optional

import numpy as np
import nesrs

class NESEmulatorClient:
    """Python client for communicating with the Rust NES emulator server via shared memory."""
    VALID_KEYS = [1, 2, 4, 8, 16, 32, 64, 128]

    def __init__(self):
        self.emu = None

    def load_rom(self, rom_path: str):
        """Load ROM."""
        self.emu = nesrs.Emulator(rom_path, True)

    def reset(self):
        """Reset emulator."""
        self.emu.reset_cpu()

    def step(self):
        """Step emulator."""
        self.emu.step_emulation()

    def set_key(self, keycode: int, pressed: bool):
        """Set key state."""
        self.emu.set_key_event(keycode, pressed)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame."""
        return np.frombuffer(self.emu.get_current_frame(), dtype=np.uint8).reshape((240,256,3))

    def get_value_at_address(self, address: int) -> int:
        """Get RAM value."""
        return self.emu.get_value_at_address(address)

    def press_key(self, key: int):
        """Press a key on the virtual joypad."""
        self.set_key(key, True)

    def release_key(self, key: int):
        """Release a key on the virtual joypad."""
        self.set_key(key, False)