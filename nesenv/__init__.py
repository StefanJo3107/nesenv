from gymnasium.envs.registration import register

register(
    id="nesenv/NESEnv-v0",
    entry_point="nesenv.envs:NESEnvironment",
)

register(
    id="nesenv/PacManEnv-v0",
    entry_point="nesenv.envs:PacManEnvironment",
)
