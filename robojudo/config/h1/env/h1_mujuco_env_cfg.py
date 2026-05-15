from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.sim_stability import SimStabilityCfg

from .h1_env_cfg import H1EnvCfg

# ===== Sim-test fall detection defaults =====
# Permissive thresholds: only catastrophic policy failures should trip this.
# Production safety_check is unaffected — this only fires when the env cfg
# enables ``sim_stability``, which the sim-test harness opts into.
_H1_UPPER_CONTACT_BODIES = [
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]


def h1_default_sim_stability_cfg() -> SimStabilityCfg:
    """Default fall-detection thresholds for the H1 (pelvis nominal at ~1.1 m)."""
    return SimStabilityCfg(
        base_body_name="pelvis",
        min_base_height=0.55,
        min_height_duration=0.6,
        max_tilt_rad=1.2,
        max_tilt_duration=0.6,
        illegal_contact_bodies=_H1_UPPER_CONTACT_BODIES,
        illegal_contact_duration=0.2,
        grace_seconds=1.0,
    )


class H1MujocoEnvCfg(H1EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    pass
