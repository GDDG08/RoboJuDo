from robojudo.environment.env_cfgs import MujocoEnvCfg

from .h1_2_env_cfg import H1_2EnvCfg


class H1_2MujocoEnvCfg(H1_2EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = True
    # ProtoMotions' MujocoSimulatorConfig.use_implicit_pd defaults True for
    # BUILT_IN_PD checkpoints (protomotions/simulator/mujoco/config.py) --
    # match it here for eval-parity with checkpoints trained/evaluated
    # under that default (e.g. h1_2_bm_dr_amass). See robojudo_prep.md.
    use_implicit_pd: bool = True
