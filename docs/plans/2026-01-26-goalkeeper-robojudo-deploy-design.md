# RoboJuDo 部署 Humanoid Goalkeeper（G1 真机）设计

日期：2026-01-26

## 1. 目标与范围
- **目标**：在 RoboJuDo 框架内接入 Humanoid Goalkeeper 策略，实现 G1 真机闭环控制；足球检测来自 `football_detectcpp` 的 DDS 话题 `detectionresults`，RoboJuDo 直接订阅并构建观测。
- **范围**：单进程 Python，包含 DDS 订阅、观测构建、策略推理、关节控制、安全与异常处理。

## 2. 关键假设
- 使用 **G1 29DoF**，与训练配置 `G129Cfg` 对齐。
- 相机外参来自 **g1_29.urdf** 的 `d435_joint`：
  - `T_torso_cam`：`xyz=(0.103, 0.01753, 0.420107)`，`rpy=(0,0,0)`。
- `torso_link ≈ pelvis`（腰部三关节变化忽略），IMU 在 pelvis。
- 策略模型为 **PyTorch checkpoint**：  
  `/home/chunyu/programs/Humanoid-Goalkeeper/legged_gym/resources/weight/goalkeeper.pt`

## 3. 总体架构
```
football_detectcpp (DDS detectionresults)
          │
          ▼
RoboJuDo 控制器订阅 → 坐标变换 → 观测构建 → ActorCritic 推理
          │                                      │
          └────────────── UnitreeEnv(低层控制) ◄─┘
```

组件划分：
- **Controller**：新增 `GoalkeeperBallDdsCtrl`，订阅 DDS 并缓存最近一次球检测。
- **Policy**：新增 `GoalkeeperPolicy`，构建 96 维 one-step 观测与 10 帧历史，加载 ActorCritic 权重并输出 29 维动作。
- **Env**：沿用 `UnitreeEnv`（unitree_sdk2py）获取真实传感器并下发控制。

## 4. DDS 订阅与数据格式
检测消息（来自 `football_detectcpp/dds/DetectionModule.idl`）：
- 话题：`detectionresults`
- `DetectionResults.results`: 多个 `DetectionResult`
  - `class_id` / `class_name` / `score` / `xyz[3]` / `offset` / `offset_fov`

策略使用 **class_id==0 (Ball)**，选最高 `score` 结果；没有有效结果时标记无球。

## 5. 坐标变换与球位置
输入 `p_cam` 为检测返回的相机坐标系 xyz（单位：米）：

1) **相机 → torso**  
`p_torso = R_tc * p_cam + t_tc`  
其中 `R_tc=I`，`t_tc=(0.103, 0.01753, 0.420107)`。

2) **torso → world**  
`p_world = torso_pos + R_torso * p_torso`

3) **world → base/pelvis**（与训练一致）  
`ball_pos = R_base^T * (p_world - torso_pos)`

注意：
- `R_torso` 来自 FK 的 `torso_quat`，无 FK 时退回 `base_quat`。
- 若 RealSense 光学坐标与检测输出存在轴差异，可增加 `R_opt_to_cam` 修正（默认单位阵）。

## 6. 观测构建（与训练一致）
one-step 观测维度 **96**：
1. `ball_pos`（3）  
2. `base_ang_vel`（3）  
3. `projected_gravity`（3）  
4. `(q - default_q)`（29）  
5. `dq`（29）  
6. `last_action`（29）

历史长度 `num_actor_history=10`，输入 ActorCritic 为 **960 维**（10 * 96）。

缩放严格按训练配置：
- `ball_pos_scale = 0.3`
- `ang_vel_scale = 0.25`
- `dof_pos_scale = 1.0`
- `dof_vel_scale = 0.05`

`last_action` 为**上一帧 policy 输出（未乘 action_scale）**。

## 7. 动作到关节目标
策略输出 `action`（29）：
```
action = clip(action, -100, 100)
target_q = default_q + action * 0.25
```
`default_q` 采用训练时 `G129Cfg.init_state.default_joint_angles`，按 G1 29DoF 顺序展开：
```
[-0.1, 0.2, 0.0, 0.3, -0.2, -0.2,
 -0.1, -0.2, 0.0, 0.3, -0.2, 0.2,
 0.0, 0.0, 0.0,
 0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.0,
 0.0, -0.5, 0.0, 1.2, 0.0, 0.0, 0.0]
```

## 8. 异常处理与安全策略
- **检测超时**：`>0.3s` 未更新，`ball_pos=0`，继续使用 `last_action`。
- **检测无效**（NaN 或全 -1）：忽略并保持上一帧有效值。
- **姿态失衡**：复用 RoboJuDo `safety_check`，倾角过大触发 `shutdown`。
- **长时间无球**：可插值回站立姿态（2~3s），防止动作漂移。

## 9. 依赖与环境
必须：
- `torch`（加载 checkpoint）
- `cyclonedds`（DDS 订阅与 IDL 类型）
- `unitree_sdk2py`（低层控制）

## 10. 测试计划
1) **DDS 订阅单测**：订阅 `detectionresults`，确认 Ball 的 `xyz` 更新。
2) **观测范围检查**：打印 one-step 96 维统计，确认缩放正确。
3) **闭环灰度测试**：无球站姿稳定性 → 有球动作响应 → 安全停机检查。

## 11. 风险与待办
- 外参可能与真机实际安装偏差，需后续标定。
- FK 依赖 mujoco 模型一致性，真机可能存在误差。
- cyclonedds Python 版本与系统 DDS 兼容性需验证。
