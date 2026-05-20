import logging
import copy
import joblib
import pickle
from pathlib import Path

import numpy as np
import onnxruntime 

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import BFMZeroPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation

logger = logging.getLogger(__name__)


@policy_registry.register
class BFMZeroPolicy(Policy):
    cfg_policy: BFMZeroPolicyCfg

    def __init__(self, cfg_policy: BFMZeroPolicyCfg, device):
        device = "cpu"
        logger.info(f"Loading onnx policy from {cfg_policy.policy_file}")
        self.onnx_policy_session = onnxruntime.InferenceSession(cfg_policy.policy_file)
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
        cfg_policy_new = cfg_policy.model_copy()
        super().__init__(cfg_policy=cfg_policy_new, device=device)
        
        self.action_scales = np.array(cfg_policy.obs_dof.action_scales)

        self.default_pos = np.array(self.cfg_policy.obs_dof.default_pos)
        
        self.action_rescale = cfg_policy.action_rescale

        self.max_timestep = self.cfg_policy.max_timestep

        self.task_type = getattr(cfg_policy, 'task_type', 'single')
        self.model_path = cfg_policy.policy_file

        self._init_task_variables()

        self.reset()


    def _prepare_policy(self):
        obs_shape = self.onnx_policy_session.get_inputs()[0].shape  # e.g. [1, 154]
        obs = np.zeros((1, obs_shape[1]), dtype=np.float32)  # Create 2D obs
        self.get_action(obs)

    def _init_task_variables(self):
        """Initialize task-specific variables based on task_type"""
        if self.task_type == "tracking":
            # Initialize tracking with multiple prompt files
            ctx_path = getattr(self.cfg_policy, 'ctx_path', None)
            if ctx_path:
                # For tracking, ctx_path is relative to BFM0 directory
                full_ctx_path = Path(self.model_path).parent / ctx_path
                logger.info(f"Loading tracking context from {full_ctx_path}")
                self.ctx = joblib.load(full_ctx_path)
                
                # Find all available tracking prompt files
                tracking_dir = Path(self.model_path).parent / "tracking_inference"
                self.available_ctx_files = sorted(tracking_dir.glob("*.pkl"))
                self.current_ctx_index = 0
                
                # Find current file index
                for i, ctx_file in enumerate(self.available_ctx_files):
                    if ctx_file.name == ctx_path:
                        self.current_ctx_index = i
                        break
                
                logger.info(f"Available tracking prompts: {[f.name for f in self.available_ctx_files]}")
                logger.info(f"Current prompt: {self.available_ctx_files[self.current_ctx_index].name}")
            else:
                raise ValueError("ctx_path must be provided for tracking task")
            
            self.t_start = getattr(self.cfg_policy, 'start', 0)
            self.t_end = getattr(self.cfg_policy, 'end', 2000)
            self.t_stop = getattr(self.cfg_policy, 'stop', 0)
            self.gamma = getattr(self.cfg_policy, 'gamma', 0.8)
            self.window_size = getattr(self.cfg_policy, 'window_size', 3)
            self.t = self.t_stop
            self.start_motion = False
            self.action_rescale = self.cfg_policy.action_rescale #5
            
        elif self.task_type == "reward":
            self.z_index = 0
            ctx_path = getattr(self.cfg_policy, 'ctx_path', None)
            if ctx_path:
                # For reward, ctx_path may be relative or absolute
                if ctx_path.startswith("../"):
                    # Relative path from BFM0 directory
                    full_ctx_path = Path(self.model_path).parent / ctx_path[3:]  # Remove "../" and use BFM0 directory
                else:
                    # Absolute path from reward_inference directory
                    full_ctx_path = Path(self.model_path).parent / "reward_inference" / ctx_path
                logger.info(f"Loading reward context from {full_ctx_path}")
                with open(full_ctx_path, "rb") as f:
                    self.z_dict_raw = pickle.load(f)
                    self.z_dict_raw_copy = copy.deepcopy(self.z_dict_raw)
                    logger.info(f"Available z_dict={list(self.z_dict_raw.keys())}")
            else:
                raise ValueError("ctx_path must be provided for reward task")
            
            # Process selected rewards filter
            selected_rewards_filter_z = getattr(self.cfg_policy, 'selected_rewards_filter_z', None)
            if selected_rewards_filter_z is None:
                raise ValueError("selected_rewards_filter_z must be provided for reward task")
            
            self.z_dict = {}
            self.selected_z_names = []
            
            # Iterate in the order of selected_rewards_filter_z
            if isinstance(selected_rewards_filter_z, list):
                for dct in selected_rewards_filter_z:
                    k = dct['reward']
                    selected_z_ids = dct['z_ids']
                    if k in self.z_dict_raw:
                        v = self.z_dict_raw[k]
                        self.z_dict[k] = []
                        for z_id in selected_z_ids:
                            if z_id < len(v):
                                self.z_dict[k].append(v[z_id])
                                self.selected_z_names.append(f"""Reward="{k}"__Z_id={z_id}""")
                                logger.info(f"""Added Reward="{k}"__Z_id={z_id} to self.z_dict""")
            
            if len(self.z_dict) == 0:
                raise ValueError("After filtering, self.z_dict is empty. Please check your selected_rewards_filter_z")
            
            self.num_selected_rewards = len(self.z_dict.keys())
            self.num_selected_z = len(self.selected_z_names)
            self.selected_z = np.concatenate([val for val in self.z_dict.values()], axis=0)
            
            logger.info(f"self.num_selected_z={self.num_selected_z}, self.selected_z.shape={self.selected_z.shape}")
            if self.num_selected_rewards == 1:
                logger.info("Only one reward is selected, make sure that is what you want")
                
        elif self.task_type == "goal":
            self.z_index = 0
            ctx_path = getattr(self.cfg_policy, 'ctx_path', None)
            if ctx_path:
                # For goal, ctx_path may be relative or absolute
                if ctx_path.startswith("../"):
                    # Relative path from BFM0 directory
                    full_ctx_path = Path(self.model_path).parent / ctx_path[3:]  # Remove "../" and use BFM0 directory
                else:
                    # Absolute path from goal_inference directory
                    full_ctx_path = Path(self.model_path).parent / "goal_inference" / ctx_path
                logger.info(f"Loading goal context from {full_ctx_path}")
                self.z_dict = joblib.load(full_ctx_path)
                self.z_dict_raw = copy.deepcopy(self.z_dict)
                logger.info(f"Available z_dict={list(self.z_dict_raw.keys())}")
            else:
                raise ValueError("ctx_path must be provided for goal task")
            
            # Process selected goals
            selected_goals = getattr(self.cfg_policy, 'selected_goals', list(self.z_dict.keys()))
            self.z_dict = {
                k: self.z_dict[k] for k in selected_goals if k in self.z_dict
            }
            
            logger.info(f"Valid z_dict contains: {list(self.z_dict.keys())} (Total = {len(self.z_dict)})")
            self.num_selected_goals = len(self.z_dict.keys())
            if self.num_selected_goals == 1:
                logger.info("Only one goal is selected, make sure that is what you want")

    def reset(self):
        self.timestep: float = self.cfg_policy.start_timestep
        self.pbar = None
        self.play_speed: float = 1.0
        self.flag_motion_done = False
        self._prepare_policy()
        
        # Reset task-specific variables
        if self.task_type == "tracking":
            self.t = self.t_stop
            self.start_motion = False
        elif self.task_type == "reward":
            self.z_index = 0
        elif self.task_type == "goal":
            self.z_index = 0


    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1 * self.play_speed
        if self.pbar and hasattr(self, 'start_motion') and self.start_motion:
            self.pbar.set(self.timestep)

        # Don't auto-start motion when max_timestep is reached
        # User needs to press "[" key to start tracking motion
        
        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[BFM_RESET_STOP_STATE]":
                    # For tracking task, reset to stop position
                    self.flag_motion_done = False

                    if self.task_type == "tracking":
                        self.t = self.t_stop
                        self.start_motion = False
                        logger.info("Policy activated, tracking at stop position")
                    elif self.task_type == "goal":
                        self.z_index = 0
                        logger.info("Policy activated, goal reset")
                        logger.info(f"Switched to goal z_index={self.z_index}")

                    elif self.task_type == "reward":
                        self.z_index = 0
                        logger.info("Policy activated, reward reset")
                        logger.info(f"Switched to reward z_index={self.z_index}")
                        

                case "[BFM_MOTION_START]":
                    # Start tracking motion (triggered by "[" key)
                    self.flag_motion_done = False

                    if self.task_type == "tracking":
                        self.start_motion = True
                        self.t = self.t_start
                        logger.info("Starting tracking motion")
                        
                case "[BFM_NEXT]":
                    # Switch to next prompt/reward/goal (triggered by "n" key)
                    if self.task_type == "tracking":
                        if hasattr(self, 'available_ctx_files') and len(self.available_ctx_files) > 0:
                            self.current_ctx_index = (self.current_ctx_index + 1) % len(self.available_ctx_files)
                            new_ctx_file = self.available_ctx_files[self.current_ctx_index]
                            logger.info(f"Loading new tracking context: {new_ctx_file.name}")
                            self.ctx = joblib.load(new_ctx_file)
                            # Reset tracking state
                            self.t = self.t_stop
                            self.start_motion = False
                    elif self.task_type == "reward":
                        if hasattr(self, 'num_selected_z') and self.num_selected_z > 0:
                            self.z_index = (self.z_index + 1) % self.num_selected_z
                            logger.info(f"Switched to reward z_index={self.z_index}")
                    elif self.task_type == "goal":
                        if hasattr(self, 'num_selected_goals') and self.num_selected_goals > 0:
                            self.z_index = (self.z_index + 1) % self.num_selected_goals
                            current_goal = list(self.z_dict.keys())[self.z_index]
                            logger.info(f"Switched to goal '{current_goal}' (z_index={self.z_index})")
                            
                case "[BFM_LAST]":
                    # Switch to last prompt/reward/goal (triggered by "m" key)
                    if self.task_type == "tracking":
                        if hasattr(self, 'available_ctx_files') and len(self.available_ctx_files) > 0:
                            self.current_ctx_index = (self.current_ctx_index - 1) % len(self.available_ctx_files)
                            new_ctx_file = self.available_ctx_files[self.current_ctx_index]
                            logger.info(f"Loading new tracking context: {new_ctx_file.name}")
                            self.ctx = joblib.load(new_ctx_file)
                            # Reset tracking state
                            self.t = self.t_stop
                            self.start_motion = False
                    elif self.task_type == "reward":
                        if hasattr(self, 'num_selected_z') and self.num_selected_z > 0:
                            self.z_index = (self.z_index - 1) % self.num_selected_z
                            logger.info(f"Switched to reward z_index={self.z_index}")
                    elif self.task_type == "goal":
                        if hasattr(self, 'num_selected_goals') and self.num_selected_goals > 0:
                            self.z_index = (self.z_index - 1) % self.num_selected_goals
                            current_goal = list(self.z_dict.keys())[self.z_index]
                            logger.info(f"Switched to goal '{current_goal}' (z_index={self.z_index})")



    def _apply_task_specific_obs(self, obs):
        """Apply task-specific observation modifications"""
        if self.task_type == "tracking":

            window = self.ctx[self.t:self.t+self.window_size] 
            discounts = self.gamma ** np.arange(len(window)) 
            discounts = discounts / np.sum(discounts)                                                                                                                                                        
            discounted_avg = np.sum(window * discounts[:, np.newaxis], axis=0)
            discounted_avg = discounted_avg / np.linalg.norm(discounted_avg, axis=-1) * np.linalg.norm(self.ctx[0])
           
            # obs is 2D [1, N], concatenate along axis=-1 and keep 2D format
            obs = np.concatenate([obs, discounted_avg[np.newaxis, :]], axis= -1).astype(np.float32)
            # Remove flatten() to keep 2D format as expected by ONNX
            

            if hasattr(self, 'start_motion') and self.start_motion and self.t < self.t_end:
                self.t += 1
                self.t = self.t % self.ctx.shape[0]
            elif hasattr(self, 'start_motion'):
                self.t = self.t_stop
                self.start_motion = False
                
        elif self.task_type == "reward":
            # Add reward-specific z vector to observation
            if hasattr(self, 'selected_z') and hasattr(self, 'z_index'):
                z_vector = self.selected_z[self.z_index]
                # z_vector is (1, 256), obs is (1, 465), both are 2D, can concatenate directly
                obs = np.concatenate([obs, z_vector], axis=-1).astype(np.float32)
            else:
                raise ValueError("selected_z or z_index not initialized for reward task")
                
        elif self.task_type == "goal":
            # Add goal-specific z vector to observation
            if hasattr(self, 'z_dict') and hasattr(self, 'z_index'):
                goal_z = list(self.z_dict.values())[self.z_index]
                obs = np.concatenate([obs, goal_z], axis=-1).astype(np.float32)
            else:
                raise ValueError("z_dict or z_index not initialized for goal task")
                
        return obs


    def get_observation(self, env_data, ctrl_data):
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        ang_vel = env_data.base_ang_vel

        # Get observation configurations from config (loaded from YAML in config file)
        default_pos = np.array(self.cfg_policy.obs_dof.default_pos)
        dof_pos_scale = self.cfg_policy.dof_pos_scale
        dof_vel_scale = self.cfg_policy.dof_vel_scale
        projected_gravity_scale = self.cfg_policy.projected_gravity_scale
        base_ang_vel_scale = self.cfg_policy.base_ang_vel_scale
        prev_actions_scale = self.cfg_policy.prev_actions_scale
        history_steps = self.cfg_policy.history_steps

        # Calculate observations according to BFM Zero observation classes
        dof_pos_minus_default = (dof_pos - default_pos) * dof_pos_scale
        dof_vel_scaled = dof_vel * dof_vel_scale
        
        # Projected gravity: rotate [0, 0, -1] by inverse of base quaternion

        projected_gravity = get_gravity_orientation(env_data.base_quat)* projected_gravity_scale

        base_ang_vel_scaled = ang_vel * base_ang_vel_scale
        
        # Previous actions - ensure correct shape
        prev_actions = self.last_action.reshape(-1) * prev_actions_scale
        
        # History observations
        # Initialize history buffers if not exists
        if not hasattr(self, 'prev_actions_history'):
            self.prev_actions_history = np.zeros((history_steps, self.num_actions))
        if not hasattr(self, 'base_ang_vel_history'):
            self.base_ang_vel_history = np.zeros((history_steps, 3))
        if not hasattr(self, 'dof_pos_minus_default_history'):
            self.dof_pos_minus_default_history = np.zeros((history_steps, self.num_dofs))
        if not hasattr(self, 'dof_vel_history'):
            self.dof_vel_history = np.zeros((history_steps, self.num_dofs))
        if not hasattr(self, 'projected_gravity_history'):
            self.projected_gravity_history = np.zeros((history_steps, 3))
        
        # Update history buffers (roll and add new data)
        self.prev_actions_history = np.roll(self.prev_actions_history, 1, axis=0)
        self.prev_actions_history[0, :] = self.last_action.reshape(-1)
        
        self.base_ang_vel_history = np.roll(self.base_ang_vel_history, 1, axis=0)
        self.base_ang_vel_history[0, :] = ang_vel
        
        self.dof_pos_minus_default_history = np.roll(self.dof_pos_minus_default_history, 1, axis=0)
        self.dof_pos_minus_default_history[0, :] = dof_pos - default_pos
        
        self.dof_vel_history = np.roll(self.dof_vel_history, 1, axis=0)
        self.dof_vel_history[0, :] = dof_vel
        
        self.projected_gravity_history = np.roll(self.projected_gravity_history, 1, axis=0)
        self.projected_gravity_history[0, :] = projected_gravity
        
        # Flatten history observations
        prev_actions_history_flat = self.prev_actions_history.reshape(-1) * 1.0
        base_ang_vel_history_flat = self.base_ang_vel_history.reshape(-1) * 0.25
        dof_pos_minus_default_history_flat = self.dof_pos_minus_default_history.reshape(-1) * 1.0
        dof_vel_history_flat = self.dof_vel_history.reshape(-1) * 1.0
        projected_gravity_history_flat = self.projected_gravity_history.reshape(-1) * 1.0
        
        # Concatenate all observations
        obs_prop = np.concatenate([
            dof_pos_minus_default,# yes
            dof_vel_scaled,# yes
            projected_gravity,# yes
            base_ang_vel_scaled,# yes
            prev_actions, # check
            prev_actions_history_flat, # check
            base_ang_vel_history_flat,
            dof_pos_minus_default_history_flat,
            dof_vel_history_flat,
            projected_gravity_history_flat
        ])

        obs = obs_prop.reshape(1, -1).astype(np.float32)
        
        # Apply BFM Zero task-specific modifications
        obs = self._apply_task_specific_obs(obs)
        
        extras = {
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],
        }
        return obs, extras
        


    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Check what inputs the model expects
        # input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        # obs is already 2D [1, N] from get_observation, so no need to expand_dims
        ort_inputs = {"actor_obs": obs.astype(np.float32)}

        action = self.onnx_policy_session.run(None, ort_inputs)[0]
        action = action.clip(-1, 1)
        action = self.action_rescale * action
        
        # Ensure last_action has correct shape for observation
        self.last_action = action.reshape(-1).copy()

        scaled_actions = action * self.action_scales

        return scaled_actions.reshape(-1)

    # def get_init_dof_pos(self) -> np.ndarray:
    #     """
    #     Return default DOF positions for BFM Zero.
    #     """
    #     return self.default_pos.copy()
