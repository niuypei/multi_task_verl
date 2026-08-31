# 多任务资源共享调度重新设计：范围与术语

> 状态：当前重新设计的已确认边界。
> 本文只记录已经评审并确认的结论；后续设计思路继续遵循“先代码核验和正确性审视，经确认后再写入文档”的流程。

## 1. 目标

重新调研和设计多任务资源共享调度子项目，覆盖 verl v0.8.0 中 actor policy rollout 的两种目标运行模式：

1. `RolloutMode.HYBRID`
2. `RolloutMode.STANDALONE`

后续所有组件、资源模型、调度协议、生命周期和权重同步分析，都必须分别验证这两种模式，不能只从
HYBRID 共卡路径推导出一套结论后直接套用到 STANDALONE。

## 2. 明确排除的模式

本轮不分析 `RolloutMode.COLOCATED`，也不把以下辅助模型场景纳入调度目标：

- GRM / Generative Reward Model / LLM-as-a-Judge；
- colocated Reward Model；
- colocated Teacher Model；
- 其他在训练 placement group 中新建独立 rollout workers 的辅助推理模型。

排除原因不是这些路径不存在，而是当前项目聚焦于需要跟随 actor policy 动态更新的 rollout 推理资源共享。
COLOCATED 在 v0.8.0 中主要服务静态或独立权重的辅助模型，其 Worker 所有权、权重语义和调用阶段与 actor
rollout 不同。若未来纳入，必须单独评审，不能默认继承 HYBRID 或 STANDALONE 的设计。

## 3. HYBRID 的本文语义

本文中的 HYBRID 严格对应 verl `RolloutMode.HYBRID`：

```text
Actor training 与 rollout 角色在同一组训练 Ray workers 中融合
  + 复用 actor training WorkerGroup
  + 使用同一批 GPU 配额
  + 训练和生成分阶段切换显存状态
  + rollout 权重跟随 actor policy 更新
```

代码事实：

```python
async def init_hybrid(self, worker_group):
    self.rollout_mode = RolloutMode.HYBRID
    self.workers = worker_group.workers[...]
    await self.launch_servers()
```

`ActorRolloutRefWorker` 在同一训练 Worker 进程中构造 actor `TrainingWorker` 和 rollout object。对于 server-based
rollout，真实 vLLM/SGLang server 仍可能包含独立 Ray Actor 或后端子进程；这不改变 HYBRID 在 verl 层面
复用训练 workers、GPU 配额和权重生命周期的语义。

## 4. STANDALONE 的本文语义

本文中的 STANDALONE 严格对应 verl `RolloutMode.STANDALONE`：

```text
Actor training 与 rollout 使用不同 Ray workers 和独立资源池
  + rollout replica 创建自己的 RayResourcePool
  + 新建 CheckpointEngineWorker actors
  + rollout server 使用独立 GPU 资源
  + 训练权重通过跨 Worker 的 Checkpoint Engine 等机制更新
```

代码事实：

```python
async def init_standalone(self):
    self.rollout_mode = RolloutMode.STANDALONE
    resource_pool_manager = ResourcePoolManager(...)
    resource_pool_manager.create_resource_pool()
    worker_group = RayWorkerGroup(resource_pool=self.resource_pool, ...)
```

普通 `LLMServerManager` 在没有传入训练 `worker_group` 时选择 `init_standalone()`。One-Step Off-Policy、
Fully Async 和独立 generation server 是当前主要调用路径。

## 5. 两种模式必须分别回答的问题

| 设计问题 | HYBRID | STANDALONE |
|---|---|---|
| 可调度资源属于谁 | 训练 WorkerGroup/任务资源池 | rollout 独立资源池 |
| replica workers 如何获得 | 从训练 workers 切分复用 | 创建 `CheckpointEngineWorker` actors |
| GPU 是否与训练共享 | 是 | 否 |
| 训练和生成能否同时运行 | 通常受同 GPU/显存互斥约束 | 资源上可以并行，算法语义另行判断 |
| 新增 replica 的 placement | 必须处理训练 workers 与 GPU 的归属 | 可以从独立 rollout 资源分配创建 |
| 权重如何到达 rollout | 同 Worker 直接更新/naive 路径为主 | 非 naive Checkpoint Engine 或后续权重存储机制 |
| 缩容如何回收 | 不能破坏训练 Worker 所有权 | 可回收独立 rollout workers/resource pool |
| 故障影响范围 | 可能同时影响训练与生成 | 主要隔离在 rollout 侧 |

表中内容是后续调研的问题边界，不代表动态扩缩方案已经确定。

## 6. 不作为部署模式的其他维度

以下配置会影响设计，但不与 HYBRID/STANDALONE 并列：

| 维度 | 示例 | 含义 |
|---|---|---|
| 请求执行模式 | sync / async rollout | 请求和 batch 如何执行 |
| 训练时序 | on-policy / one-step / fully async | 样本版本和训练更新关系 |
| 推理 backend | vLLM / SGLang / TensorRT-LLM | server 和权重加载实现 |
| replica 内部拓扑 | TP / DP / PP / PD disaggregation | 单个 replica 的资源占用和内部通信 |
| 权重同步 backend | naive / NCCL / NIXL / Mooncake 等 | trainer 到 rollout 的权重数据通道 |

后续分析应在 HYBRID/STANDALONE 两种目标模式下按需展开这些维度，避免把它们混成新的部署模式。

## 7. 代码依据

| 事实 | verl v0.8.0 位置 |
|---|---|
| `RolloutMode` 三个枚举值及注释 | `verl/workers/rollout/replica.py:54-67` |
| HYBRID 复用训练 worker handles | `verl/workers/rollout/replica.py:131-141` |
| STANDALONE 创建独立资源池和 workers | `verl/workers/rollout/replica.py:189-225` |
| COLOCATED 创建同 PG 独立进程，当前排除 | `verl/workers/rollout/replica.py:159-187` |
| manager 依据是否传入 `worker_group` 选择 HYBRID/STANDALONE | `verl/workers/rollout/llm_server.py:297-325` |
| sync actor 将 `actor_rollout_wg` 传给 manager | `verl/trainer/main_ppo_sync.py:711-714` |
| One-Step 不传 WorkerGroup，进入 STANDALONE | `verl/experimental/one_step_off_policy/ray_trainer.py:170-196` |
| Fully Async 显式进入 STANDALONE 初始化分支 | `verl/experimental/fully_async_policy/fully_async_rollouter.py:203-225` |

## 8. 已确认决策记录

### D1：目标模式范围

```text
IN SCOPE:
  - RolloutMode.HYBRID
  - RolloutMode.STANDALONE

OUT OF SCOPE:
  - RolloutMode.COLOCATED
  - GRM / Reward / Teacher colocated deployment
```

确认日期：2026-08-19。
