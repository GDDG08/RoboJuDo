# Policy

**Policy** is the component that controls the robot. It receives the `env_data` from the environment, `ctrl_data` from the controller, organize the observation and infer the action fo robot.

## [Policy](#policy)

`Policy` is the base class for all policies. It defines the interface for the policy, as in [base_policy.py](../robojudo/policy/base_policy.py)

We provide the following policies:
- [UnitreePolicy](#policy--unitreepolicy)
- [AMOPolicy](#policy--amopolicy)
- [BeyondMimicPolicy](#policy--beyondmimicpolicy)
- [H2HStudentPolicy](#policy--h2hstudentpolicy)

## [Policy](#policy) > [UnitreePolicy](#policy--unitreepolicy)

`UnitreePolicy` is the policy that controls the robot using the [Unitree official policy](https://github.com/unitreerobotics/unitree_rl_gym). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [unitree_policy.py](../robojudo/policy/unitree_policy.py)

To control the robot using `UnitreePolicy`, you can refer `_get_commands()` to set commands:

`commands`:
- `commands[0]`, [-1, 1], control the robot to walk forward and backward
- `commands[1]`, [-1, 1], control the robot to walk left and right
- `commands[2]`, [-1, 1], control the robot to turn left and right

for instance, use `JoystickCtrl` to control:

```python
def _get_commands(self, ctrl_data: dict) -> list[float]:
    commands = np.zeros(3)
    for key in ctrl_data.keys():
        if key in ["JoystickCtrl", "UnitreeCtrl"]:
            axes = ctrl_data[key]["axes"]
            lx, ly, rx, ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

        commands[0] = command_remap(ly, self.commands_map[0])
        commands[1] = command_remap(lx, self.commands_map[1])
        commands[2] = command_remap(rx, self.commands_map[2])
    return commands
```

## [Policy](#policy) > [AMOPolicy](#policy--amopolicy)

`AMOPolicy` is the policy that controls the robot using the [AMO](https://github.com/OpenTeleVision/AMO). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [amo_policy.py](../robojudo/policy/amo_policy.py)

To control the robot using `AMOPolicy`, you can refer `_get_commands()` to set commands:

`commands`:
- `commands[0]`, [-1, 1], control the robot to walk forward and backward
- `commands[1]`, [-1, 1], control the robot to turn left and right
- `commands[2]`, [-1, 1], control the robot to walk left and right
- `commands[3]`, [-0.5, 0.8], control the robot torso height
- `commands[4]`, [-1.57, 1.57], control the robot torso yaw
- `commands[5]`, [-0.52, 1.57], control the robot torso pitch
- `commands[6]`, [-0.7, 0.7], control the robot torso roll

You can apply your own controller to control the robot using `AMOPolicy`. Just set the `commands` in `_get_comands()`

## [Policy](#policy) > [BeyondMimicPolicy](#policy--beyondmimicpolicy)

`BeyondMimicPolicy` is the policy that controls the robot using the [BeyondMimic](https://github.com/han-xudong/beyondmimic). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [beyondmimic_policy.py](../robojudo/policy/beyondmimic_policy.py)

`BeyondMimicPolicy` is controlled by `BeyondMimicCtrl`. We don't recommand to override its `_get_command()`. Instead, we suggest you set it with cfg file:
[`G1BeyondMimicPolicyCfg`](../robojudo/config/g1/policy/g1_beyondmimic_policy_cfg.py):
 - `policy_name`: The name of the policy. We provive `2025-09-03_21-00-31_Box` for test. You should put your policy in `assets/deploy_models/g1/beyondmimic`
 - `without_state_estimator`: Whether to use state estimator. Default is `False`.
 - `use_motion_from_model`: Whether to use motion from model. Default is `False`. If `False`, you need to provide motion file through `BeyondMimicCtrl: motion_name`.
 - `use_modelmeta_config`: Whether to use modelmeta config. Default is `True`. If `False`, the policy will use your `env`'s meta config, like `kp`, `kd`, `action_scale`, which could lead to unpredictable behavior
.

 You can refer [g1_beyondmimic_ctrl_cfg.py](../robojudo/config/g1/ctrl/g1_beyondmimic_ctrl_cfg.py), [beyondmimic_ctrl.py](../robojudo/controller/beyondmimic_ctrl.py) and [g1_beyondmimic_policy_cfg.py](../robojudo/config/g1/policy/g1_beyondmimic_policy_cfg.py) for details

example:
```python
# Run 2025-09-03_21-00-31_Box
class G1BeyondmimicCfg(RlPipelineCfg):
    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="2025-09-03_21-00-31_Box",
        use_motion_from_model=True,
    ),

# Run a policy without state estimator and different motion
class G1BeyondmimicCfg(RlPipelineCfg):
    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
        BeyondMimicCtrlCfg(
            motion_name="Dance"
        ),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="2025-09-03_21-00-31_Box",
        without_state_estimator=True,
        use_motion_from_model=False,
    ),
```

## [Policy](#policy) > [HugWBCPolicy](#policy--hugwbcpolicy)

`HugWBCPolicy` is the policy that controls the robot using the [HugWBC](https://github.com/apexrl/HugWBC). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [hugwbc_policy.py](../robojudo/policy/hugwbc_policy.py)

🥺Will release soon.

## [Policy](#policy) > [H2HStudentPolicy](#policy--h2hstudentpolicy)

`H2HStudentPolicy` is the policy that controls the robot using the [H2HStudent](https://github.com/LeCAR-Lab/human2humanoid). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [h2hstudent_policy.py](../robojudo/policy/h2hstudent_policy.py)

`H2HStudentPolicy` is controlled by `G1MotionCtrl`. You can refer [g1_motion_ctrl_cfg](../robojudo/config/g1/ctrl/g1_motion_ctrl_cfg.py) to control it.

