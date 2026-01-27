import numpy as np
from box import Box

from robojudo.config.g1.policy.g1_goalkeeper_policy_cfg import G1GoalkeeperPolicyCfg
from robojudo.policy.goalkeeper_policy import GoalkeeperPolicy


def test_goalkeeper_obs_shape():
    cfg = G1GoalkeeperPolicyCfg(load_checkpoint=False)
    policy = GoalkeeperPolicy(cfg_policy=cfg, device="cpu")

    env_data = Box(
        dof_pos=np.zeros(29),
        dof_vel=np.zeros(29),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        base_ang_vel=np.zeros(3),
        torso_pos=np.zeros(3),
        torso_quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    ctrl_data = Box({"GoalkeeperBallDdsCtrl": {"ball_xyz_cam": np.array([1.0, 0.0, 0.0]), "ball_valid": True}})
    obs, extras = policy.get_observation(env_data, ctrl_data)
    assert obs.shape == (96,)
