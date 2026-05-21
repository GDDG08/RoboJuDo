"""Compare the G1 torso (secondary) IMU against the FK-derived torso pose.

Connects to a real G1 and, every control step, reads:
  - the raw torso IMU from ``rt/secondary_imu`` (quaternion + gyroscope)
  - the FK-derived torso pose (from the pelvis IMU + joint encoders)

Both are drawn in mid-air in a MuJoCo viewer so the discrepancy is visible:
  - orientation: the torso_link forward axis
  - angular velocity: gyro vector in the world frame

Colors:
  IMU -> orientation: blue,  angular velocity: cyan
  FK  -> orientation: red,   angular velocity: orange

born_place_align is forced off so the raw IMU and FK share the same world frame.

Usage::

    python scripts/test_torso_imu.py [--net-if eth0] [--sdk cpp|py]
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from robojudo.environment import env_registry
from robojudo.tools.kinematics import MujocoKinematics
from robojudo.tools.tool_cfgs import ForwardKinematicCfg

TORSO_NAME = "torso_link"
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # xyzw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-if", type=str, default="eth0", help="network interface to the robot")
    parser.add_argument("--sdk", choices=["cpp", "py"], default="cpp", help="unitree backend")
    return parser.parse_args()


def get_raw_torso_imu(env):
    """Return (quat_xyzw, gyro) of the raw torso IMU, or (None, None) if no sample yet."""
    state = getattr(env, "torso_imu_state", None)  # UnitreeEnv (sdk2py)
    if state is None:
        robot_state = getattr(env, "robot_state", None)  # UnitreeCppEnv
        state = getattr(robot_state, "torso_imu_state", None) if robot_state is not None else None
    if state is None:
        return None, None

    quat_wxyz = np.asarray(state.quaternion, dtype=np.float32)
    gyro = np.asarray(state.gyroscope, dtype=np.float32)
    # default-constructed IMU is identity (w=1, xyz=0) -> treat as "no sample yet"
    if quat_wxyz[0] == 1.0 and not quat_wxyz[1:].any():
        return None, None
    return quat_wxyz[[1, 2, 3, 0]], gyro


def main():
    args = parse_args()

    cfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv" if args.sdk == "cpp" else "UnitreeEnv",
        unitree=G1UnitreeCfg(net_if=args.net_if, enable_torso_imu=True),
        born_place_align=False,  # keep raw IMU and FK in the same world frame
        update_with_fk=False,    # FK is run by this script directly, not the env
    )

    env_class = env_registry.get(cfg.env_type)
    env = env_class(cfg_env=cfg)

    # standalone kinematics with a viewer; the env's own kinematics has no viz
    viz_kine = MujocoKinematics(
        cfg=ForwardKinematicCfg(
            xml_path=cfg.xml,
            debug_viz=True,
            kinematic_joint_names=cfg.dof.joint_names,
        )
    )
    viz = viz_kine.visualizer

    print("[test_torso_imu] running, Ctrl-C to stop")
    while True:
        env.update()

        base_pos = env.base_pos if env.base_pos is not None else np.array([0.0, 0.0, 0.8])
        # FK from pelvis IMU + joint encoders; also renders the robot
        fk_info = viz_kine.forward(
            joint_pos=env.dof_pos,
            base_pos=base_pos,
            base_quat=env.base_quat,
            joint_vel=env.dof_vel,
            base_ang_vel=env.base_ang_vel,
        )
        fk_torso = fk_info[TORSO_NAME]
        fk_quat = np.asarray(fk_torso["quat"], dtype=np.float32)  # xyzw
        fk_ang_vel = np.asarray(fk_torso["ang_vel"], dtype=np.float32)  # world frame (cvel)
        ori_anchor = np.asarray(fk_torso["pos"], dtype=np.float32)
        av_anchor = ori_anchor + np.array([0.0, 0.0, 0.4], dtype=np.float32)

        imu_quat, imu_gyro = get_raw_torso_imu(env)

        # --- orientation arrows (torso forward axis) ---
        viz.draw_arrow(ori_anchor, fk_quat, [0.3, 0, 0], color=[1, 0, 0, 1], id=0)  # FK: red
        if imu_quat is not None:
            viz.draw_arrow(ori_anchor, imu_quat, [0.3, 0, 0], color=[0, 0, 1, 1], id=1)  # IMU: blue

        # --- angular-velocity arrows (world frame, identity root_quat) ---
        viz.draw_arrow(av_anchor, IDENTITY_QUAT, fk_ang_vel, color=[1, 0.5, 0, 1], scale=0.5, id=2)  # FK: orange
        if imu_quat is not None:
            imu_ang_vel_world = sRot.from_quat(imu_quat).apply(imu_gyro)
            viz.draw_arrow(av_anchor, IDENTITY_QUAT, imu_ang_vel_world, color=[0, 1, 1, 1], scale=0.5, id=3)  # IMU: cyan

        if imu_quat is not None:
            quat_err_deg = np.degrees((sRot.from_quat(imu_quat) * sRot.from_quat(fk_quat).inv()).magnitude())
            print(f"quat err: {quat_err_deg:6.2f} deg | gyro err: {np.linalg.norm(imu_ang_vel_world - fk_ang_vel):.3f} rad/s")
        else:
            print("waiting for torso IMU sample on rt/secondary_imu ...")

        time.sleep(0.02)


if __name__ == "__main__":
    main()
