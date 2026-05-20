from robojudo.policy.policy_cfgs import UnitreeMjlabVelocityPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig, convert_29dof_to_23dof

class G1UnitreeMjlabDoF(DoFConfig):
    '''
    '''
    joint_names: list[str] = [
        'left_hip_pitch_joint', 
        'left_hip_roll_joint', 
        'left_hip_yaw_joint', 
        'left_knee_joint', 
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint', 
        'right_hip_pitch_joint', 
        'right_hip_roll_joint', 
        'right_hip_yaw_joint', 
        'right_knee_joint', 
        'right_ankle_pitch_joint', 
        'right_ankle_roll_joint', 
        'waist_yaw_joint', 
        'waist_roll_joint', 
        'waist_pitch_joint', 
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 
        'left_elbow_joint', 
        'left_wrist_roll_joint', 
        'left_wrist_pitch_joint', 
        'left_wrist_yaw_joint', 
        'right_shoulder_pitch_joint', 
        'right_shoulder_roll_joint', 
        'right_shoulder_yaw_joint', 
        'right_elbow_joint', 
        'right_wrist_roll_joint', 
        'right_wrist_pitch_joint', 
        'right_wrist_yaw_joint'
        ]
    default_pos: list[float] | None = [-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0,  0,0,0,  0.35,0.18,0,0.87,0,0,0, 0.35,-0.18,0,0.87,0,0,0]
    stiffness: list[float] | None = [40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 28.5, 28.5,
        14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8]

    damping: list[float] | None = [2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 1.8, 1.8,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1]
    action_scale: list[float] = [0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.35, 0.44, 0.44,
                     0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07, 0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07]
    action_offset: list[float] = [-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0,  0,0,0,  0.35,0.18,0,0.87,0,0,0, 0.35,-0.18,0,0.87,0,0,0]
    
    

class G1UnitreeMjlab23DoF(DoFConfig):
    '''
    修改内容：
    - 23DoF版本移除两个手腕的两个电机pitch、yaw共四个以及两个腰部电机roll, pitch，一共六个。
    - remove: wrist_yaw*2, wrist_pitch*2, waist_roll_joint, waist_pitch_joint
    '''
    # Get 29DoF configuration
    _dof_29 = G1UnitreeMjlabDoF()
    
    # Convert to 23DoF
    joint_names: list[str] = convert_29dof_to_23dof(_dof_29.joint_names, _dof_29.joint_names)
    default_pos: list[float] | None = convert_29dof_to_23dof(_dof_29.joint_names, _dof_29.default_pos)
    stiffness: list[float] | None = convert_29dof_to_23dof(_dof_29.joint_names, _dof_29.stiffness)
    damping: list[float] | None = convert_29dof_to_23dof(_dof_29.joint_names, _dof_29.damping)
    action_scale: list[float] = convert_29dof_to_23dof(_dof_29.joint_names, _dof_29.action_scale)
    action_offset: list[float] = convert_29dof_to_23dof(_dof_29.joint_names, _dof_29.action_offset)

class G1UnitreeMjlabVelocity29DoFPolicyCfg(UnitreeMjlabVelocityPolicyCfg):
    """
    Unitree MJLab Velocity Policy (29-DoF variant).
    Uses G1UnitreeMjlabDoF (29 joints) for full-DoF observation/action.
    Required for the `demo_29dof` ONNX model which expects 96-dim input.
    """
    obs_dof: DoFConfig = G1UnitreeMjlabDoF()

    action_dof: DoFConfig = obs_dof

    history_obs_dims: dict[str, int] = {
        "ang_vel": 3,
        "gravity": 3,
        "commands": 3,
        "dof_pos": obs_dof.num_dofs,
        "dof_vel": obs_dof.num_dofs,
        "actions": action_dof.num_dofs,
    }

    max_cmd: list[float] = [1.0, 0.5, 1.5]
    commands_map: list[list[float]] = [
        [-0.6, 0.0, 0.9],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


class G1UnitreeMjlabVelocityPolicyCfg(UnitreeMjlabVelocityPolicyCfg):
    """
    参考 UnitreeWoGaitPolicyCfg
    将UnitreeWoGaitPolicyCfg+G1UnitreeWoGaitPolicyCfg合并
    /home/hero/Projects/Robotics/Sim2Real/RoboJuDo/robojudo/policy/policy_cfgs.py
    /home/hero/Projects/Robotics/Sim2Real/RoboJuDo/robojudo/config/g1/policy/g1_unitree_policy_cfg.py
    
    Unitree Velocity policy configuration from unitree_rl_mjlab.
    
    This configuration uses ONNX model for velocity control,
    directly loading and using the original unitree_rl_mjlab model.
    Default obs_dof is 23-DoF (G1UnitreeMjlab23DoF); for the 29-DoF
    variant (demo_29dof model) use G1UnitreeMjlabVelocity29DoFPolicyCfg.
    
    Features:
    - Direct ONNX model inference
    - Original unitree_rl_mjlab velocity control parameters
    - Keyboard-based velocity commands (WASD+QE)
    - Training configuration compatibility
    """
    # obs_dof: DoFConfig = G1UnitreeMjlabDoF()
    obs_dof: DoFConfig = G1UnitreeMjlab23DoF()

    action_dof: DoFConfig = obs_dof
    
    history_obs_dims: dict[str, int] = {
        "ang_vel": 3,
        "gravity": 3, 
        "commands": 3,
        "dof_pos": obs_dof.num_dofs,
        "dof_vel": obs_dof.num_dofs,
        "actions": action_dof.num_dofs,
    }

    max_cmd: list[float] = [1.0, 0.5, 1.5]
    commands_map: list[list[float]] = [
        [-0.6, 0.0, 0.9],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]

