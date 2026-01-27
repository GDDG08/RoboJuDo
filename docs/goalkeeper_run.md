# RoboJuDo Goalkeeper 运行说明（G1 真机 + DDS）

本说明面向 **仅部署 Goalkeeper** 的场景，不包含 PHC/TWIST/KungfuBot/H2H 等模块。

## 1. 前置条件

- **机器人**：Unitree G1（29DoF），网线连接本机。
- **DDS 发布端**：`football_detectcpp` 正在发布 `detectionresults` 话题。
- **系统依赖**：
  - `unitree_sdk2`（建议放在 `/home/chunyu/programs/unitree_sdk2`）
  - CycloneDDS C 库（**版本 0.10.2**）
- **Python 依赖**：
  - `cyclonedds==0.10.2`（**必须安装，否则控制器无法启动**）
  - `torch`（由 `pip install -e .` 安装）
- **模型文件**：
  - `/home/chunyu/programs/Humanoid-Goalkeeper/legged_gym/resources/weight/goalkeeper.pt`

> 说明：即使 DDS 发布端没启动（订阅不到数据），程序仍可运行，但 `ball_valid=False`，球位置会置零。

## 2. 环境配置（新机器必做）

### 2.1 创建 Python 环境并安装 RoboJuDo
```bash
conda create -n robojudo python=3.11 -y
conda activate robojudo
cd /home/chunyu/programs/RoboJuDo/.worktrees/goalkeeper-integration
python -m pip install -e .
```

### 2.2 初始化 submodule（worktree 里需要）
```bash
git submodule update --init --recursive packages/unitree_cpp
```

### 2.3 绑定 unitree_sdk2 并安装 unitree_cpp
```bash
export UNITREE_SDK2_ROOT=/home/chunyu/programs/unitree_sdk2
export CMAKE_LIBRARY_PATH=$UNITREE_SDK2_ROOT/lib/x86_64:$UNITREE_SDK2_ROOT/thirdparty/lib/x86_64
export CMAKE_INCLUDE_PATH=$UNITREE_SDK2_ROOT/include:$UNITREE_SDK2_ROOT/thirdparty/include:$UNITREE_SDK2_ROOT/thirdparty/include/ddscxx
export LD_LIBRARY_PATH=$UNITREE_SDK2_ROOT/lib/x86_64:$UNITREE_SDK2_ROOT/thirdparty/lib/x86_64:$LD_LIBRARY_PATH
export CXXFLAGS="-I$UNITREE_SDK2_ROOT/thirdparty/include"

python -m pip install -e packages/unitree_cpp
```

### 2.4 安装 cyclonedds（Python）
```bash
python -m pip install cyclonedds==0.10.2
```

## 3. 配置检查

### 3.1 运行配置
配置名：`g1_goalkeeper_real`  
文件：`robojudo/config/g1/g1_cfg.py`

默认使用：
- `UnitreeCppEnv`
- 控制器：`UnitreeCtrl` + `GoalkeeperBallDdsCtrl`
- 策略：`GoalkeeperPolicy`

### 3.2 网络接口
修改 `robojudo/config/g1/g1_cfg.py` 里的 `net_if`：

```
g1_goalkeeper_real.env.unitree.net_if = "eth0"
```

### 3.3 策略与模型
文件：`robojudo/config/g1/policy/g1_goalkeeper_policy_cfg.py`

需要确认：
- `checkpoint_path` 指向你的 `goalkeeper.pt`
- `goalkeeper_repo_path` 指向 Humanoid-Goalkeeper 仓库根目录
- 相机外参 `camera_R_tc` / `camera_t_tc` 是否与实际安装一致

### 3.4 DDS 订阅参数
文件：`robojudo/config/g1/ctrl/g1_goalkeeper_ball_dds_ctrl_cfg.py`

可调整：
- `domain_id`
- `topic_name`（默认 `detectionresults`）
- `timeout_s`（默认 0.3s）

## 4. 运行步骤

1) 启动 DDS 检测发布端  
确保 `football_detectcpp` 正在发布 `detectionresults`。

2) 启动 RoboJuDo Goalkeeper 管线  
在仓库根目录执行：

```bash
python scripts/run_pipeline.py -c g1_goalkeeper_real
```

## 5. 运行期说明

- **Start 开关**：程序启动后默认站立，按手柄 `Start` 开始策略；再次按 `Start` 停止策略并回到站立。
- **无检测数据**：会保持 `ball_valid=False`，球位置置零，策略仍可运行。  
- **无 cyclonedds**：控制器无法启动（这是预期行为）。  
- **停止运行**：直接 `Ctrl+C` 退出即可。

## 6. 常见问题

**1) `ModuleNotFoundError: cyclonedds`**  
需要安装 `cyclonedds` Python 包，并确保系统有 CycloneDDS C 库。  
若 pip 安装失败，请检查系统是否已安装 CycloneDDS。

**2) `unitree_cpp` 编译失败**  
通常是 `UNITREE_SDK2_ROOT` 或 include/lib 路径不对。  
请确认 `UNITREE_SDK2_ROOT/include` 与 `UNITREE_SDK2_ROOT/thirdparty/include` 中存在头文件。

**3) 订阅不到数据**  
确认：
- DDS 发布端已启动  
- 话题名是 `detectionresults`  
- `domain_id` 一致
