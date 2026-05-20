import time
import numpy as np
import logging
from queue import Empty, Queue

logger = logging.getLogger(__name__)

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import BFMKeyboardCtrlCfg, BFMJoystickCtrlCfg
from robojudo.controller.utils.keyboard import KeyboardThread
from robojudo.controller.joystick_ctrl import JoystickCtrl


@ctrl_registry.register
class BFMKeyboardCtrl(Controller):
    cfg_ctrl: BFMKeyboardCtrlCfg

    def __init__(self, cfg_ctrl: BFMKeyboardCtrlCfg, env=None, **kwargs):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, **kwargs)

        self.event_queue = Queue(maxsize=100)
        self.keyboard_thread = KeyboardThread(self.event_queue)
        self.keyboard_thread.start()

        # Track pressed keys for continuous movement
        self.pressed_keys = set()
        
        # BFM Zero specific state
        self.get_ready_state = False
        self.start_motion = False
        
        # Velocity commands for movement
        self.lin_vel_command = np.zeros((1, 2))  # [forward/backward, left/right]
        self.ang_vel_command = np.zeros((1, 1))  # [angular velocity]
        
        # KP level for debugging
        self.kp_level = 1.0
        
        # Virtual axes for keyboard movement control
        self.axes = {
            "LeftX": 0.0,    # left/right movement (a/d)
            "LeftY": 0.0,    # forward/backward movement (w/s)  
            "RightX": 0.0,   # turning (q/e)
            "RightY": 0.0,   # unused
        }

        self.reset()

    def reset(self):
        self.pressed_keys.clear()
        self.axes = {
            "LeftX": 0.0,
            "LeftY": 0.0, 
            "RightX": 0.0,
            "RightY": 0.0,
        }
        # Reset BFM state
        self.get_ready_state = False
        self.start_motion = False
        self.lin_vel_command = np.zeros((1, 2))
        self.ang_vel_command = np.zeros((1, 1))
        
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Empty:
                break

    def get_events(self):
        events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except Empty:
                break
        return events

    def update_axes(self):
        """Update virtual axes based on currently pressed keys"""
        # Reset axes
        self.axes["LeftX"] = 0.0
        self.axes["LeftY"] = 0.0
        self.axes["RightX"] = 0.0
        
        # Forward/backward (w/s)
        if "w" in self.pressed_keys:
            self.axes["LeftY"] = 1.0
        elif "s" in self.pressed_keys:
            self.axes["LeftY"] = -1.0
            
        # Left/right strafe (a/d)
        if "a" in self.pressed_keys:
            self.axes["LeftX"] = -1.0
        elif "d" in self.pressed_keys:
            self.axes["LeftX"] = 1.0
            
        # Turn left/right (q/e)
        if "q" in self.pressed_keys:
            self.axes["RightX"] = -1.0
        elif "e" in self.pressed_keys:
            self.axes["RightX"] = 1.0

    def get_data(self):
        events = self.get_events()
        
        # Update pressed keys state
        for event in events:
            if event["type"] == "keyboard":
                if event["pressed"]:
                    self.pressed_keys.add(event["name"])
                else:
                    self.pressed_keys.discard(event["name"])
        
        # Update virtual axes
        self.update_axes()
        
        return {
            "keyboard_event": events,
            "axes": self.axes.copy(),
            "button_event": [],  # Empty for compatibility with joystick interface
            # BFM Zero specific data
            "bfm_state": {
                "get_ready_state": self.get_ready_state,
                "start_motion": self.start_motion,
                "lin_vel_command": self.lin_vel_command.copy(),
                "ang_vel_command": self.ang_vel_command.copy(),
                "kp_level": self.kp_level,
            }
        }

    def process_triggers(self, ctrl_data):
        commands = []
        if len(self.triggers) == 0:
            return ctrl_data, commands

        for event in ctrl_data["keyboard_event"]:
            if event["type"] == "keyboard" and not event["pressed"]:  # trigger when key is released
                command = self.triggers.get(event["name"], None)
                if command is not None:
                    # Handle BFM Zero specific commands
                    self._handle_bfm_command(event["name"], command)
                    commands.append(command)
                    # remove event after triggered
                    ctrl_data["keyboard_event"].remove(event)

        return ctrl_data, commands

    def _handle_bfm_command(self, key, command):
        """Handle BFM Zero specific keyboard commands"""

        if command == "[BFM_MOTION_START]":
            logger.info("Starting motion")
            self.start_motion = True
            


@ctrl_registry.register
class BFMJoystickCtrl(JoystickCtrl):
    cfg_ctrl: BFMJoystickCtrlCfg

    def __init__(self, cfg_ctrl: BFMJoystickCtrlCfg, env=None, **kwargs):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, **kwargs)
        
        # BFM Zero specific state
        self.get_ready_state = False
        self.start_motion = False

    def get_data(self):
        data = super().get_data()
        
        # Add BFM Zero specific state
        data["bfm_state"] = {
            "get_ready_state": self.get_ready_state,
            "start_motion": self.start_motion,
        }
        
        return data

    def process_triggers(self, ctrl_data):
        commands = []
        if len(self.triggers) == 0:
            return ctrl_data, commands

        for event in ctrl_data["button_event"]:
            if event["type"] == "button" and not event["pressed"]:  # trigger when button is released
                command = self.triggers.get(event["name"], None)
                if command is not None:
                    # Handle BFM Zero specific commands
                    self._handle_bfm_command(event["name"], command)
                    commands.append(command)
                    # remove event after triggered
                    ctrl_data["button_event"].remove(event)

        return ctrl_data, commands

    def _handle_bfm_command(self, button, command):
        """Handle BFM Zero specific joystick commands"""
        if command == "[BFM_POLICY_ACTIVATE]":
            logger.info("Policy activate command (policy is always active)")
            
        elif command == "[BFM_ACTIONS_ZERO]":
            logger.info("Actions set to zero (not implemented - policy always active)")
            
        elif command == "[BFM_INIT_STATE]":
            logger.info("Setting to init state")
            self.get_ready_state = True
            
        elif command == "[BFM_MOTION_START]":
            logger.info("Starting motion")
            self.start_motion = True
            
        elif command == "[BFM_RESET_STOP_STATE]":
            logger.info("Resetting to stop state")
            self.start_motion = False
            
        elif command == "[BFM_NEXT_REWARD_GOAL]":
            logger.info("Switch to next reward/goal")
            # This would need to be connected to the policy
