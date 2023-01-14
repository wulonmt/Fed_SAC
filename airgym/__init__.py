from gym.envs.registration import register

register(
    id="airsim-car-cont-action-sample-v0", entry_point="airgym.envs:AirSimCarEnvContAction",
)
