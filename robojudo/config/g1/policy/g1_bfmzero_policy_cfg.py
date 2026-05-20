from pathlib import Path
import yaml

from pydantic import ConfigDict, model_validator

from robojudo.policy.policy_cfgs import BFMZeroPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig,convert_29dof_to_23dof

def load_yaml_config(path_config: str | Path):
    """Load YAML configuration from config.yaml"""
    yaml_path = Path(path_config)
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def map_joint_config(joint_names, config_dict, default_value=0.0):
    """Map configuration values to joint names using pattern matching"""
    result = []
    for joint_name in joint_names:
        matched = False
        for pattern, value in config_dict.items():
            if pattern.replace('.*', '') in joint_name:
                result.append(value)
                matched = True
                break
        if not matched:
            result.append(default_value)
    return result


class G1BFMZeroDoF(DoFConfig):
    """G1 BFM Zero DoF configuration with YAML-based parameters"""
    
    joint_names: list[str] = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
              'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
              'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
              'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
              'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
              'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
              'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
    
    default_pos: list[float] | None = None
    stiffness: list[float] | None = None
    damping: list[float] | None = None
    torque_limits: list[float] | None = None
    action_scales: list[float] | None = None

    def load_from_yaml(self, yaml_config: dict):
        self.default_pos = map_joint_config(
            self.joint_names, yaml_config['default_joint_pos'], 0.0
        )
        self.stiffness = map_joint_config(
            self.joint_names, yaml_config['joint_kp'], 0.0
        )
        self.damping = map_joint_config(
            self.joint_names, yaml_config['joint_kd'], 0.0
        )
        # self.torque_limits = map_joint_config(
        #     self.joint_names, yaml_config['joint_effort_limit']
        # )
        self.action_scales = map_joint_config(
            self.joint_names, yaml_config['action_scale'], 1.0
        )



class G1BFMZero23DoF(G1BFMZeroDoF):
    """G1 BFM Zero 23DoF configuration"""
    # Get 29DoF configuration
    _dof_29 = G1BFMZeroDoF()
    
    # Convert to 23DoF
    joint_names: list[str] = convert_29dof_to_23dof(_dof_29.joint_names, _dof_29.joint_names)

    def load_from_yaml(self, yaml_config: dict):
        # We need to map 29dof first, then convert to 23dof
        full_default_pos = map_joint_config(self._dof_29.joint_names, yaml_config['default_joint_pos'], 0.0)
        full_stiffness = map_joint_config(self._dof_29.joint_names, yaml_config['joint_kp'], 0.0)
        full_damping = map_joint_config(self._dof_29.joint_names, yaml_config['joint_kd'], 0.0)
        full_action_scales = map_joint_config(self._dof_29.joint_names, yaml_config['action_scale'], 1.0)

        self.default_pos = convert_29dof_to_23dof(self._dof_29.joint_names, full_default_pos)
        self.stiffness = convert_29dof_to_23dof(self._dof_29.joint_names, full_stiffness)
        self.damping = convert_29dof_to_23dof(self._dof_29.joint_names, full_damping)
        self.action_scales = convert_29dof_to_23dof(self._dof_29.joint_names, full_action_scales)


class G1BFMZeroPolicyCfg(BFMZeroPolicyCfg):
    robot: str = "g1"

    obs_dof: G1BFMZeroDoF = G1BFMZeroDoF()
    action_dof: G1BFMZeroDoF = None

    @model_validator(mode='after')
    def load_config_from_yaml(self) -> 'G1BFMZeroPolicyCfg':
        if self.action_dof is None:
            self.action_dof = self.obs_dof
        yaml_config = load_yaml_config(self.config_file)
        if yaml_config:
            self.obs_dof.load_from_yaml(yaml_config)
        return self
    action_beta: float = 1.0
    
    # YAML-based configurations
    action_rescale: int = 5
    
    # Observation configurations from YAML
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 1.0
    projected_gravity_scale: float = 1.0
    base_ang_vel_scale: float = 0.25
    prev_actions_scale: float = 1.0
    history_steps: int = 4
    
    # BFM Zero Task Configuration
    task_type: str = "single"  # Options: "single", "tracking", "reward", "goal"

    # For tracking task
    start: int = 0
    end: int = 2000
    stop: int = 0
    ctx_path: str = "tracking_inference/zs_7.pkl"  # Path to context file relative to model directory
    gamma: float = 0.8
    window_size: int = 3

    # Override action_scales from the DoF configuration
    # action_scales: list[float] = G1BFMZeroDoF().action_scales or [1.0] * len(G1BFMZeroDoF().joint_names)

    # # For reward task
    # selected_rewards_filter_z: list = None

    # # For goal task
    # selected_goals: list = None


class G1BFMZero23DoFPolicyCfg(G1BFMZeroPolicyCfg):
    """BFM Zero Policy for 23DoF"""
    obs_dof: G1BFMZero23DoF = G1BFMZero23DoF()
    action_dof: G1BFMZero23DoF = None

    @model_validator(mode='after')
    def load_config_from_yaml(self) -> 'G1BFMZero23DoFPolicyCfg':
        if self.action_dof is None:
            self.action_dof = self.obs_dof
        yaml_config = load_yaml_config(self.config_file)
        if yaml_config:
            self.obs_dof.load_from_yaml(yaml_config)
        return self


class G1BFMZeroTracking23DoFPolicyCfg(G1BFMZero23DoFPolicyCfg):
    """BFM Zero 23DoF Policy for tracking tasks"""
    task_type: str = "tracking"
    ctx_path: str = "tracking_inference/zs_7.pkl"
    # train_method: str = "23dof_low_20260407_182514"
    train_method: str = "23dof_260411"
    start: int = 0
    end: int = 2000
    stop: int = 0
    gamma: float = 0.8
    window_size: int = 3


class G1BFMZeroTrackingPolicyCfg(G1BFMZeroPolicyCfg):
    """BFM Zero Policy for tracking tasks"""
    task_type: str = "tracking"
    ctx_path: str = "tracking_inference/zs_7.pkl"
    start: int = 0
    end: int = 2000
    stop: int = 0
    gamma: float = 0.8
    window_size: int = 3


class G1BFMZeroRewardPolicyCfg(G1BFMZeroPolicyCfg):
    """BFM Zero Policy for reward-based tasks"""
    task_type: str = "reward"
    ctx_path: str = "../reward_inference/reward_locomotion.pkl"
    train_method: str = "low"

    selected_rewards_filter_z: list = [
        {"reward": "move-ego-low0.6-0-0.7", "z_ids": [0]},
        {"reward": "move-ego-90-0.3", "z_ids": [0]},
        {"reward": "move-ego-0-0", "z_ids": [0]},
        {"reward": "spin-arms-5-l-l", "z_ids": [0]},
    ]

class G1BFMZeroReward23DoFPolicyCfg(G1BFMZero23DoFPolicyCfg):
    """BFM Zero 23DoF Policy for reward-based tasks"""
    task_type: str = "reward"
    # train_method: str = "23dof_low_20260407_182514"
    train_method: str = "23dof_260411"
    ctx_path: str = "../reward_inference/reward_locomotion.pkl"
    selected_rewards_filter_z: list = [
        {"reward": "move-ego-low0.6-0-0.7", "z_ids": [0]},
        {"reward": "move-ego-90-0.3", "z_ids": [0]},
        {"reward": "move-ego-0-0", "z_ids": [0]},
        {"reward": "spin-arms-5-l-l", "z_ids": [0]},
    ]

class G1BFMZeroGoalPolicyCfg(G1BFMZeroPolicyCfg):
    """BFM Zero Policy for goal-based tasks"""
    task_type: str = "goal"
    ctx_path: str = "../goal_inference/goal_reaching.pkl"
    train_method: str = "low"

    # selected_goals: list = [
    #     "fallAndGetUp1_subject4_2193",
    #     "dance1_subject3_505", 
    #     "fightAndSports1_subject1_252",
    #     "walk2_subject1_2588",
    # ]

class G1BFMZeroGoal23DoFPolicyCfg(G1BFMZero23DoFPolicyCfg):
    """BFM Zero 23DoF Policy for goal-based tasks"""
    task_type: str = "goal"
    ctx_path: str = "../goal_inference/goal_reaching.pkl"
    # train_method: str = "23dof_low_20260407_182514"
    train_method: str = "23dof_260411"

