from robojudo.policy.policy_cfgs import GoalkeeperPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1GoalkeeperDoF(DoFConfig):
    joint_names: list[str] = [
        *[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        *[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        *["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        *[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        *[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    ]

    default_pos: list[float] | None = [
        *[-0.1, 0.2, 0.0, 0.3, -0.2, -0.2],
        *[-0.1, -0.2, 0.0, 0.3, -0.2, 0.2],
        *[0.0, 0.0, 0.0],
        *[0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.0],
        *[0.0, -0.5, 0.0, 1.2, 0.0, 0.0, 0.0],
    ]


class G1GoalkeeperPolicyCfg(GoalkeeperPolicyCfg):
    robot: str = "g1"

    obs_dof: DoFConfig = G1GoalkeeperDoF()
    action_dof: DoFConfig = obs_dof
