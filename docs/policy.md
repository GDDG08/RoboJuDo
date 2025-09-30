# Policy

**Policy** is the component that controls the robot. It receives the `env_data` from the environment, `ctrl_data` from the controller, organize the observation and infer the action fo robot.

## [Policy](#policy)

`Policy` is the base class for all policies. It defines the interface for the policy, as in [base_policy.py](../robojudo/policy/base_policy.py)

---

We provide the following policies:
- [UnitreePolicy](#policy--unitreepolicy)
- [AMOPolicy](#policy--amopolicy)
- [H2HStudentPolicy](#policy--h2hstudentpolicy)
- [BeyondMimicPolicy](#policy--beyondmimicpolicy)

## [Policy](#policy) > [UnitreePolicy](#policy--unitreepolicy)

`UnitreePolicy` is the policy that controls the robot using the [Unitree official policy](https://github.com/unitreerobotics/unitree_rl_gym). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [unitree_policy.py](../robojudo/policy/unitree_policy.py)

To control the robot using `UnitreePolicy`, you can refer `_get_commands()`:

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
            break
    return commands
```

## [Policy](#policy) > [AMOPolicy](#policy--amopolicy)

`AMOPolicy` is the policy that controls the robot using the [AMO](https://github.com/OpenTeleVision/AMO). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [amo_policy.py](../robojudo/policy/amo_policy.py)

To control the robot using `AMOPolicy`, you can refer `_get_commands()`:

`commands`:
- `commands[0]`, [-1, 1], control the robot to walk forward and backward
- `commands[1]`, [-1, 1], control the robot to turn left and right
- `commands[2]`, [-1, 1], control the robot to walk left and right
- `commands[3]`, [-0.5, 0.8], control the robot torso height
- `commands[4]`, [-1.57, 1.57], control the robot torso yaw
- `commands[5]`, [-0.52, 1.57], control the robot torso pitch
- `commands[6]`, [-0.7, 0.7], control the robot torso roll

You can apply your own controller to control the robot using `AMOPolicy`. Just set the `commands` in `_get_comands()`


## [Policy](#policy) > [H2HStudentPolicy](#policy--h2hstudentpolicy)

`H2HStudentPolicy` is the policy that controls the robot using the [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid). It is a subclass of `Policy` and implements the interface defined in `Policy`.

> PHC Submodule is needed for motionlib control. check README#setup.

script: [h2hstudent_policy.py](../robojudo/policy/h2hstudent_policy.py)

`H2HStudentPolicy` is controlled by `MotionCtrl`. check code [motion_ctrl.py](../robojudo/controller/motion_ctrl.py).

For motion source:
- `Unitree H1`: Simply use the motion retargeting pipeline from [human2humanoid](https://github.com/LeCAR-Lab/human2humanoid).
- `Unitree G1`: As not officially supported, we use the PHC pipeline. Our submodule patch enables 29dof G1. Check [unitree_g1_29dof_fitting.yaml](../third_party/phc/phc/data/cfg/robot/unitree_g1_29dof_fitting.yaml).

You can refer to `g1_h2h` config in [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) for more details.


## [Policy](#policy) > [HugWBCPolicy](#policy--hugwbcpolicy)

`HugWBCPolicy` is the policy that controls the robot using the [HugWBC](https://github.com/apexrl/HugWBC). It is a subclass of `Policy` and implements the interface defined in `Policy`.

script: [hugwbc_policy.py](../robojudo/policy/hugwbc_policy.py)

🥺Will release soon.


## [Policy](#policy) > [BeyondMimicPolicy](#policy--beyondmimicpolicy)

`BeyondMimicPolicy` is the policy that controls the robot using the [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking). It is a subclass of `Policy` and implements the interface defined in `Policy`.

We support both `G1FlatEnvCfg` and `G1FlatWoStateEstimationEnvCfg`. 
For motion source, you could use the motion inside onnx policy, or use `BeyondmimicCtrl` with npz files.

script: [beyondmimic_policy.py](../robojudo/policy/beyondmimic_policy.py)

[`BeyondMimicPolicyCfg`](../robojudo/policy/policy_cfgs.py): check example at [G1BeyondMimicPolicyCfg](../robojudo/config/g1/policy/g1_beyondmimic_policy_cfg.py):
 - `policy_name`: The name of the policy. We provive `Jump_wose` for test. You should put your policy in `assets/models/g1/beyondmimic`
 - `without_state_estimator`: Weather policy is `WoStateEstimation`. Default is `True`.
 - `use_modelmeta_config`: Whether to use modelmeta config. Default is `True`. If `False`, the policy will use config in your `BeyondMimicPolicyCfg`.
 - `use_motion_from_model`: Whether to use motion in the onnx model. Default is `True`. If `False`, you need to enable `BeyondMimicCtrl`.
.

 You can refer to `g1_beyondmimic` and `g1_beyondmimic_with_ctrl` in [g1_cfg.py](../robojudo/config/g1/g1_cfg.py) for details.

## [Policy](#policy) > [AsapPolicy](#policy--asappolicy)

`AsapPolicy` is the policy that controls the robot using the [ASAP](https://github.com/LeCAR-Lab/ASAP). It is a subclass of `Policy` and implements the interface defined in `Policy`.

RoboJuDo support both `deepmimic` and `decoupled_locomotion` of the official repo, implemeted as `AsapPolicy` and `AsapLocoPolicy`.

We fully reproduced the original repository, including keyboard and joystick mapping:

- press `i` to make the robot the initial position
- press `o` to emergence stop the robot
- press `=` to switch between tapping and walking for the locomotion policy
- press `w/a/s/d` to control the linear velocity
- press `q/e` to control the angular velocity
- press `z` to set all commands to zero

script: [asap_policy.py](../robojudo/policy/asap_policy.py)

> For your convenience, `CR7_level1` checkpoint is included, you can ran sim2sim with `g1_asap` config in [g1_asap_cfg.py](../robojudo/config/g1/g1_asap_cfg.py).

You can add more models to `assets/models/g1/asap/mimic`. Any model in the official repo and [RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy) should work.

This example highlights the advantages of RoboJudo:
- Modular code & config with easy implementation and strong readability
- Flexible policy switching, with interpolation support.
- Convenient external controller processing

You can refer to `g1_asap` and `g1_asap_loco` in [g1_cfg.py](../robojudo/config/g1/g1_asap_cfg.py) for test and details.

