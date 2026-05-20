from robojudo.policy.policy_cfgs import GentlePolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

# G1 robot joint configuration for gentle policy
G1_GENTLE_DOF_CFG = DoFConfig(
        joint_names=[
            "left_hip_pitch_joint",
            "right_hip_pitch_joint", 
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint"
        ],
        default_pos=[
            -0.28,
            -0.28,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.35,
            0.35,
            -0.23,
            -0.23,
            0.16,
            -0.16,
            0.0,
            0.0,
            0.0,
            0.0,
            0.87,
            0.87,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        stiffness=[
            40.17923847137318,
            40.17923847137318,
            40.17923847137318,
            99.09842777666113,
            99.09842777666113,
            28.50124619574858,
            40.17923847137318,
            40.17923847137318,
            28.50124619574858,
            99.09842777666113,
            99.09842777666113,
            14.25062309787429,
            14.25062309787429,
            28.50124619574858,
            28.50124619574858,
            14.25062309787429,
            14.25062309787429,
            28.50124619574858,
            28.50124619574858,
            14.25062309787429,
            14.25062309787429,
            14.25062309787429,
            14.25062309787429,
            14.25062309787429,
            14.25062309787429,
            16.77832748089279,
            16.77832748089279,
            16.77832748089279,
            16.77832748089279
        ],
        damping=[
            2.5578897650279457,
            2.5578897650279457,
            2.5578897650279457,
            6.3088018534966395,
            6.3088018534966395,
            1.814445686584846,
            2.5578897650279457,
            2.5578897650279457,
            1.814445686584846,
            6.3088018534966395,
            6.3088018534966395,
            0.907222843292423,
            0.907222843292423,
            1.814445686584846,
            1.814445686584846,
            0.907222843292423,
            0.907222843292423,
            1.814445686584846,
            1.814445686584846,
            0.907222843292423,
            0.907222843292423,
            0.907222843292423,
            0.907222843292423,
            0.907222843292423,
            0.907222843292423,
            1.06814150219,
            1.06814150219,
            1.06814150219,
            1.06814150219
        ]
    )   

# Gentle Policy configuration class for G1 robot
class G1GentlePolicyCfg(GentlePolicyCfg):
    def __init__(self, **kwargs):
        # Default configuration
        default_config = {
            "robot": "g1",
            "obs_dof": G1_GENTLE_DOF_CFG,
            "action_dof": G1_GENTLE_DOF_CFG,
            
            # Policy specific settings
            "policy_name": "policy_latest",
            # Action settings (matching the JS config)
            "action_scale": [
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # legs & waist
                0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0,  # knees & ankles  
                0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # arms
                1.0, 1.0, 1.0, 1.0  # wrists
            ],
            "action_clip": 10.0,
            "action_beta": 1.0,
            
            # History and prediction settings

            "joint_pos_steps": [0, 1, 2, 3, 4, 8],
            "future_steps": [0, 2, 4, 8, 16],
            "prev_actions_steps": 3,
            
            # Compliance settings
            "compliance_enabled": False,
            "compliance_threshold": 10.0,
            
            # Motion tracking settings
            # "tracking_enabled": True,
            "transition_steps": 100,
            
            # Control frequency
            "freq": 50
        }
        
        # Merge with any provided kwargs
        default_config.update(kwargs)
        super().__init__(**default_config)
