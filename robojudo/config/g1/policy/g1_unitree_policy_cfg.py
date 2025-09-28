from robojudo.policy.policy_cfgs import UnitreePolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1UnitreeDoF(DoFConfig):
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
    ]

    default_pos: list[float] | None = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]

    stiffness: list[float] | None = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]

    damping: list[float] | None = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]

    torque_limits: list[float] | None = [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50]


class G1UnitreePolicyCfg(UnitreePolicyCfg):
    robot: str = "g1"
    policy_name: str = "policy_lstm_1"

    obs_dof: DoFConfig = G1UnitreeDoF()
    action_dof: DoFConfig = obs_dof
