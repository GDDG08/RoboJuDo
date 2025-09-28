from robojudo.policy.policy_cfgs import H2HPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1_21_H2HDoF(DoFConfig):
    joint_names: list[str] = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    default_pos: list[float] | None = [
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    stiffness: list[float] | None = [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        200,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
    ]

    damping: list[float] | None = [
        5,
        5,
        5,
        5,
        2,
        2,
        5,
        5,
        5,
        5,
        2,
        2,
        6,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]


class G1H2HPolicyCfg(H2HPolicyCfg):
    robot: str = "g1"

    # policy_name: str = "25_05_26_16-36-04_OmniH2O_STUDENT_g1_walk_21dofmodel_8500"
    # policy_name: str = "25_05_26_16-28-43_OmniH2O_STUDENT_g1_walk_21dofmodel_17000"
    # policy_name: str = "25_05_26_21-00-36_OmniH2O_STUDENT_g1_walk_21dofmodel_34500"
    # policy_name: str = "25_05_27_14-07-31_OmniH2O_STUDENT_g1_walk_21dofmodel_4000"
    # policy_name: str = "25_05_27_14-07-31_OmniH2O_STUDENT_g1_walk_21dofmodel_12500"
    policy_name: str = "h2h"

    obs_dof: DoFConfig = G1_21_H2HDoF()
    action_dof: DoFConfig = obs_dof

    # ======= POLICY SPECIFIC CONFIGURATION =======
    pass
