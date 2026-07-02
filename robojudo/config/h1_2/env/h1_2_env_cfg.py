"""H1_2 environment config for RoboJuDo.

Joint order, stiffness, and damping below are copied verbatim from the
ONNX YAML sidecar produced by ProtoMotions'
``deployment/export_bm_tracker_onnx.py`` for the H1_2 smoke checkpoint
(``results/h1_2_smoke1/last.ckpt`` -> ``unified_pipeline.yaml``), so this
config's DOF order/gains are provenance-matched to the exported policy.
``default_pos`` is all-zeros (no explicit standing-pose joint-angle table
was found in protomotions/robot_configs/h1_2.py; the tracker policy itself
overrides obs/action DOF config from the ONNX YAML at runtime anyway, and
approximates a standing default pose separately -- see
``ProtoMotionsTrackerPolicy._DEFAULT_JOINT_POS_BY_ROBOT["h1_2"]``).
"""

from robojudo.config import ASSETS_DIR
from robojudo.environment.env_cfgs import EnvCfg
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg


class H1_2_27DoF(DoFConfig):
    joint_names: list[str] = [
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    default_pos: list[float] | None = [0.0] * 27

    stiffness: list[float] | None = [
        118.43525280630205, 118.43525280630205, 118.43525280630205,
        157.91367040840274, 78.95683520420137, 39.478417602100684,
        118.43525280630205, 118.43525280630205, 118.43525280630205,
        157.91367040840274, 78.95683520420137, 39.478417602100684,
        118.43525280630205,
        19.739208801050342, 19.739208801050342, 19.739208801050342,
        19.739208801050342, 19.739208801050342, 19.739208801050342,
        19.739208801050342, 19.739208801050342, 19.739208801050342,
        19.739208801050342, 19.739208801050342, 19.739208801050342,
        19.739208801050342, 19.739208801050342,
    ]

    damping: list[float] | None = [
        7.539822368399999, 7.539822368399999, 7.539822368399999,
        10.0530964912, 5.0265482456, 2.5132741228,
        7.539822368399999, 7.539822368399999, 7.539822368399999,
        10.0530964912, 5.0265482456, 2.5132741228,
        7.539822368399999,
        1.2566370614, 1.2566370614, 1.2566370614, 1.2566370614,
        1.2566370614, 1.2566370614, 1.2566370614, 1.2566370614,
        1.2566370614, 1.2566370614, 1.2566370614, 1.2566370614,
        1.2566370614, 1.2566370614,
    ]

    # effort_limit (N*m) per joint, from protomotions/robot_configs/h1_2.py
    # ControlConfig.override_control_info (per-joint-family regex table).
    # The exported ONNX YAML sidecar's control.effort_limits was null, so
    # this is sourced directly from the ProtoMotions robot config instead
    # (needed because RoboJuDo's MujocoEnv.step() torque-clips unconditionally
    # and crashes with a TypeError if torque_limits is None).
    torque_limits: list[float] | None = [
        200, 200, 200, 300, 60, 40,
        200, 200, 200, 300, 60, 40,
        200,
        40, 40, 18, 18, 19, 19, 19,
        40, 40, 18, 18, 19, 19, 19,
    ]


class H1_2EnvCfg(EnvCfg):
    xml: str = (ASSETS_DIR / "robots/h1_2/h1_2_box_feet.xml").as_posix()

    dof: DoFConfig = H1_2_27DoF()

    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = True
    torso_name: str = "torso_link"
