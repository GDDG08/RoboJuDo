"""Compare the FK-derived torso pose against the final env torso value on a real G1.

Every control step we look at two things for ``torso_link``:
  - ``env.fk_info[torso]`` -- the pure FK estimate (pelvis IMU + joint encoders)
  - ``env.torso_quat`` / ``env.torso_ang_vel`` -- the value the env actually exposes
    to policies, which is the real torso (secondary) IMU once this fix is in.

Before the fix these two are identical (the env just forwarded FK); after the fix
the env value comes from ``rt/secondary_imu``, so the gap between them is exactly
what the fix changes. Both are drawn in the MuJoCo viewer from the torso:
  - orientation: torso forward axis -- FK: red, env: blue
  - angular velocity (world frame): FK: orange, env: cyan

born_place_align is forced off so FK and the IMU share the same world frame.

Usage::

    python tests/test_torso_imu.py [--net-if eth0] [--sdk cpp|py]
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from robojudo.environment import env_registry
from robojudo.tools.tool_cfgs import ForwardKinematicCfg

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-if", type=str, default="eth0", help="network interface to the robot")
    parser.add_argument("--sdk", choices=["cpp", "py"], default="cpp", help="unitree backend")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv" if args.sdk == "cpp" else "UnitreeEnv",
        unitree=G1UnitreeCfg(net_if=args.net_if, enable_torso_imu=True),
        born_place_align=False,  # keep FK and the IMU in the same world frame
        update_with_fk=True,     # populate env.fk_info for the comparison
    )
    # render the robot through the env's own kinematics
    cfg.forward_kinematic = ForwardKinematicCfg(
        xml_path=cfg.xml,
        debug_viz=True,
        kinematic_joint_names=cfg.dof.joint_names,
    )

    env_class = env_registry.get(cfg.env_type)
    env = env_class(cfg_env=cfg)
    viz = env.kinematics.visualizer
    torso_name = env._torso_name

    print("[test_torso_imu] running, Ctrl-C to stop")
    while True:
        env.update()  # runs FK (renders robot) and reads the torso IMU

        fk_torso = env.fk_info[torso_name]
        fk_quat = np.asarray(fk_torso["quat"], dtype=np.float32)  # xyzw, world
        fk_ang_vel = np.asarray(fk_torso["ang_vel"], dtype=np.float32)  # world (cvel)
        ori_anchor = np.asarray(fk_torso["pos"], dtype=np.float32)
        av_anchor = ori_anchor + np.array([0.0, 0.0, 0.4], dtype=np.float32)

        env_quat = env.torso_quat  # IMU truth after the fix (== FK before it)
        env_ang_vel = env.torso_ang_vel  # body-frame gyro when from the IMU
        env_ang_vel_world = sRot.from_quat(env_quat).apply(env_ang_vel)

        # --- orientation arrows (torso forward axis) ---
        viz.draw_arrow(ori_anchor, fk_quat, [0.3, 0, 0], color=[1, 0, 0, 1], id=0)  # FK: red
        viz.draw_arrow(ori_anchor, env_quat, [0.3, 0, 0], color=[0, 0, 1, 1], id=1)  # env: blue

        # --- angular-velocity arrows (world frame, identity root_quat) ---
        viz.draw_arrow(av_anchor, IDENTITY_QUAT, fk_ang_vel, color=[1, 0.5, 0, 1], scale=0.5, id=2)  # FK: orange
        viz.draw_arrow(av_anchor, IDENTITY_QUAT, env_ang_vel_world, color=[0, 1, 1, 1], scale=0.5, id=3)  # env: cyan

        quat_err_deg = np.degrees((sRot.from_quat(env_quat) * sRot.from_quat(fk_quat).inv()).magnitude())
        gyro_err = np.linalg.norm(env_ang_vel_world - fk_ang_vel)
        print(f"quat err: {quat_err_deg:6.2f} deg | gyro err: {gyro_err:.3f} rad/s")

        time.sleep(0.02)


if __name__ == "__main__":
    main()
