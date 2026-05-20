import numpy as np
import torch
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class G1UnitreeMjlabVelocityPolicy(Policy):
    """
    Unitree Velocity policy using ONNX model from unitree_rl_mjlab.
    
    This policy directly loads and uses ONNX models for velocity control,
    matching the original unitree_rl_mjlab deployment approach.
    
    Features:
    - Direct ONNX model inference
    - Original unitree_rl_mjlab velocity control parameters
    - Keyboard-based velocity commands (WASD+QE)
    - Training configuration compatibility
    """
    
    cfg_policy: "G1UnitreeMjlabVelocityPolicyCfg"

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

        # Initialize basic attributes without calling super().__init__
        self.cfg_policy = cfg_policy
        self.device = device
        self.policy_file = cfg_policy.policy_file
        self.commands_map = self.cfg_policy.commands_map

        # Load ONNX model
        self._load_onnx_model()
        
        self.default_dof_pos = np.array(self.cfg_policy.obs_dof.default_pos, dtype=np.float32)

        
        print(f"✓ Initialized default_dof_pos: {self.default_dof_pos[:5]}...")  # DEBUG
        
        # Add cfg_obs_dof for compatibility
        self.cfg_obs_dof = cfg_policy.obs_dof
        self.cfg_action_dof = cfg_policy.action_dof
        self.default_pos = self.default_dof_pos  # Alias for compatibility
        
        print(f"✓ Loaded Unitree ONNX Velocity policy: {self.cfg_policy.model_dir}/policy.onnx")
        
        # Velocity control parameters
        self.max_linear_vel = self.cfg_policy.max_cmd[0]
        self.max_lateral_vel = self.cfg_policy.max_cmd[1] 
        self.max_angular_vel = self.cfg_policy.max_cmd[2]
        
        self.reset()
    

    def _load_onnx_model(self):
        """Load ONNX model using ONNX Runtime."""
        model_path = self.policy_file

        # Setup ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if hasattr(self.device, 'type') and self.device.type == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.ort_session = ort.InferenceSession(
            model_path,providers=providers)
        
        # Get input/output names
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        
        print(f"ONNX Model inputs: {self.input_names}")
        print(f"ONNX Model outputs: {self.output_names}")

    def reset(self):
        """Reset policy state."""
        self.timestep = 0
        self.last_action = np.zeros(self.cfg_policy.action_dof.num_dofs, dtype=np.float32)
        # unitree
        self._init_history(np.zeros(self.history_obs_size))
        # wogait
        # history_obs_dims = self.cfg_policy.history_obs_dims
        # default_history = [np.zeros(dim, dtype=np.float32) for dim in history_obs_dims.values()]
        # self._init_history(default_history)

    def _get_commands(self, ctrl_data):
        """
        Process velocity commands from keyboard/joystick.
        """
        commands = np.zeros(3)
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                if "axes" in ctrl_data[key]:
                    axes = ctrl_data[key]["axes"]
                    lx, ly, rx, ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

                    commands[0] = command_remap(ly, self.commands_map[0])
                    commands[1] = command_remap(lx, self.commands_map[1])
                    commands[2] = command_remap(rx, self.commands_map[2])
                    break

            if key in ["KeyboardCtrl"]:
                keys = ctrl_data[key]["keyboard_event"]
                for event in keys:
                    if event["type"] == "keyboard":
                        value = event["pressed"] * 1.
                        match event["name"]:
                            case "w":
                                commands[0] = command_remap(value, self.commands_map[0])
                            case "s":
                                commands[0] = command_remap(-value, self.commands_map[0])
                            case "a":
                                commands[1] = command_remap(-value, self.commands_map[1])
                            case "d":
                                commands[1] = command_remap(value, self.commands_map[1])
                            case "e":
                                commands[2] = command_remap(value, self.commands_map[2])
                            case "q":
                                commands[2] = command_remap(-value, self.commands_map[2])
                break

        return commands

    # def compute_obs(self, env_data, ctrl_data):
    def get_observation(self, env_data, ctrl_data):

        """
        Compute observations for ONNX model with debug output.
        """
        # Get velocity commands
        commands = self._get_commands(ctrl_data)
        
        # Build observation (matching unitree_rl_mjlab format)
        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs = np.concatenate([
            env_data.base_ang_vel * self.cfg_policy.obs_scales.ang_vel,
            gravity_orientation * self.cfg_policy.obs_scales.gravity,
            commands * self.cfg_policy.obs_scales.command * np.array(self.cfg_policy.max_cmd),
            (env_data.dof_pos - self.default_dof_pos) * self.cfg_policy.obs_scales.dof_pos,
            env_data.dof_vel * self.cfg_policy.obs_scales.dof_vel,
            self.last_action,
        ])

        extras = {
            "commands": commands,
            # "velocity_command": commands,
        }
        
        return obs, extras

    def post_step_callback(self, commands):
        """Required abstract method implementation - pipeline only passes commands."""
        self.timestep += 1  # 更新步数用于调试
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:

        # Prepare input for ONNX model
        obs_tensor = obs.astype(np.float32)
        
        # Run ONNX inference
        with torch.no_grad():
            ort_inputs = {self.input_names[0]: obs_tensor[None, :]}  # Add batch dimension
            actions_tensor = self.ort_session.run(self.output_names, ort_inputs)
        
        # Get action from output
        action = actions_tensor[0][0]#numpy().squeeze()  # Remove batch dimension
        # action_beta: float = 1.0  # action smoothing factor 1的时候不变
        action = (1 - self.action_beta) * self.last_action + self.action_beta * action
        self.last_action = action.copy()
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)

        # Apply action scale from deploy.yaml (exact values)
        if hasattr(self.cfg_policy, 'action_scale'):
            action_scale = np.array(self.cfg_policy.obs_dof.action_scale)
            # action_offset = np.array(self.cfg_policy.action_offset)
            # action = action * action_scale + action_offset
            action = action * action_scale

        return action
