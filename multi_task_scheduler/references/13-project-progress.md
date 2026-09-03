# 多任务资源共享调度项目工作进展

> 状态：持续更新。
>
> 更新日期：2026-09-01。
>
> 本文只记录当前目标、已完成工作、有效结论、未决问题和下一步。长期原则与分析标准见仓库根目录
> `AGENTS.md`。

## 1. 当前代码与仓库基线

| 项目 | 当前状态 |
|---|---|
| 调度仓库 | `/Users/nyp/Documents/multi_task_verl` |
| 调度仓库分支 | `main` |
| 本轮开始时本地 HEAD | `9ad57e7`，`docs: separate replica bootstrap from native sync` |
| 本轮已先推送基线 | `9ad57e7`，`origin/main` 已同步 |
| 远端 | `git@github.com:niuypei/multi_task_verl.git` |
| verl 源码 | `/Users/nyp/Documents/verl` |
| verl 实际 describe | `v0.9.0-1-g88512193` |
| verl commit | `88512193` |
| 历史分析基线 | verl `v0.8.0`，已归档到 `references/archive/verl-v0.8.0` |
| 模拟器 | `/Users/nyp/Documents/multi-rl-task-scheduler` |

本轮已统一归档为参考资料的分析文档：

- `11-training-worker-dict-ray-actor-creation-call-chain.md`；
- `12-python-class-instance-call-and-rayclasswithinitargs.md`；
- `14-verl-v0.8-v0.9-hybrid-standalone-differences.md`；
- `15-verl-v0.9-hybrid-colocate-async-partial-rollout-interrupt-resume.md`；
- `16-verl-v0.9-dynamic-replica-view-and-reference-audit.md`；
- `17-verl-v0.9-dynamic-replica-scaling-timing-and-mutual-exclusion.md`；
- 本文及 `references/` 目录整理；
- 仓库根目录 `AGENTS.md`。

正式工作文档已拆分为：

- `multi_task_scheduler/多RL任务资源共享调度对接VERL - 架构及组件.md`；
- `multi_task_scheduler/[WIP] 多RL任务资源共享调度对接VERL - 关键流程.md`。

其余分析文档位于 `multi_task_scheduler/references/`。本轮不修改上述正式文档。工作区中的本地 Mermaid CLI 环境和
Excalidraw 草稿属于工具或未定稿资料，除非用户明确要求，否则不提交。

## 2. 当前项目目标

项目计划以独立子仓方式对接 verl，实现多个 RL 任务之间的 rollout GPU 动态共享，并向社区提交 RFC。

当前阶段目标是：

1. 以 verl v0.9.0 当前代码为准，完整理解 HYBRID 和 STANDALONE 的资源、进程、Actor 和调用链。
2. 明确 `GroupScheduler`、`TaskRunner`、`MultiTaskLLMServerManager`、LB 和 Checkpoint Engine 的边界。
3. 识别动态捐赠、借用、回收、参数同步和 partial rollout 所需的原生扩展点与必要改造点。
4. 先完成代码事实分析和方案评审，再更新正式 RFC，最后进入实现。

当前不以直接实现动态调度为目标，也不在尚未评审时提交侵入式 verl 修改。

## 3. 已确认的总体设计方向

### 3.1 控制关系

```text
GroupScheduler 1:N TaskRunner

GroupScheduler
  → 维护全局 task/node/GPU/replica 视图
  → 生成捐赠、受赠和回收策略
  → 通过 TaskRunner 下发命令

TaskRunner
  → 持有 GroupScheduler ActorHandle
  → 注册、注销和上报任务资源
  → 在本 controller 进程调用 MultiTaskLLMServerManager
```

不引入额外通信 Actor。

### 3.2 初始化与共享

- 初始 replica 数量由任务自身配置，GroupScheduler 不参与初始规模分配。
- 任务启动后将本任务资源注册进 GroupScheduler。
- 实时资源共享发生在训练运行阶段。
- donor 原生 replica 不销毁，执行 sleep 后保留，方便回收时唤醒。
- borrower 根据 GroupScheduler 给出的 node ID、GPU ID 创建临时推理实体。
- 不能默认复用 donor 的 ResourcePool、Placement Group 和 resource bundle。
- 最新确认：donor 不向 borrower 传递或临时出借 ResourcePool、PG、bundle、RayWorkerGroup、worker、
  `CheckpointEngineWorker`、`ServerAdapter` 或 server handle。borrowed replica 从创建开始完全由 borrower manager、
  borrower 参数同步组件和 borrower LB 管理；GroupScheduler 只维护 node/GPU lease 与 fencing。

### 3.3 三种资源视图

当前设计必须同时维护：

1. verl/Ray 原生资源视图；
2. 推理 LB server 视图；
3. GroupScheduler node ID/GPU ID 全局物理视图。

跨任务创建会使三种视图出现差异。GroupScheduler 是跨任务物理资源关系的事实来源，但仍需补充原子更新、失败回滚和
Ray 原生资源冲突规避方案。

### 3.4 参数同步

- 新 replica 的 `BOOTSTRAPPING` 与 verl 原生参数同步是两个独立操作：bootstrap 由 runtime-ready 事件触发，只把该 replica
  追平到任务当前已发布的 serving version；原生同步仍按 verl 原有 hook/频率发布下一版本并更新全部 effective replica。
- 两者的数据面必须经 task-local weight-version gate 串行。bootstrap 已开始时，原生同步
  请求可按原时机到达并 pending，但 snapshot cut、process-group 构建和传权必须等 bootstrap 完成 CE effective membership 与
  LB `ROUTABLE` 提交或失败回滚。
- bootstrap 不递增任务参数版本、不 reset staleness、不 abort 或重复同步已有 replica。数据源必须是不可变的
  `PublishedWeightSnapshot(V_serving)`，不能把可能已经领先的 trainer live weights 标成当前 serving version。
- borrower replica 完成 bootstrap、进入 CE effective membership 并通过最终 LB commit 后才能接流；gate 释放后的下一次
  原生 sync snapshot 必须包含它。
- HYBRID partial borrowed replica 可以跨过 `on_sample_end()`、training、H4 native sync 并保留到下一 step；`on_sample_end()`
  只冻结 lifecycle snapshot 并执行 abort/sleep，不隐含 CE/LB REMOVE、runtime DESTROY 或 GS lease 归还。
- H4 前必须在 task-local membership/weight-version gate 下冻结 immutable effective snapshot，包含 fixed native 与当前有效 borrowed
  replica；同步期间成员不变。HYBRID naive 固有路径只覆盖 fixed `actor_wg`，因此 true-new borrowed endpoint 仍需要 mixed-subset
  联合发布，这是 GAP，不能因 replica 在 `on_sample_end()` 被 sleep 而忽略。
- GroupScheduler 不持有 replica/worker/adapter runtime handles，也不调用 Checkpoint Engine 或改变原生 hook；它只通过 borrower
  `TaskRunner` 下发规模命令，由 borrower 任务执行 bootstrap、更新有效集合和提交路由状态。
- 原生 CE 只能覆盖它实际持有 replica/worker handles 的对象，不能从 LB server 列表自动推导覆盖范围。
- borrowed replica 的同步 endpoint 必须由 borrower 创建并拥有，不能重绑定或临时复用 donor
  `CheckpointEngineWorker`/`ServerAdapter`。
- 当前 CE 没有 target-only bootstrap + 指定已发布版本重放接口，HYBRID naive 也只覆盖固定 `actor_wg`；两者都是 GAP。
- 不能假设使用 donor PG 创建 `BorrowedCheckpointEngineWorker`；该动态 Actor 创建问题仍待 v0.9.0 代码级设计。

## 4. 已完成的研究和文档

### 4.1 v0.8.0 历史研究

以下内容已经完成并归档：

- HYBRID 启动流程和完整 PPO 迭代；
- 进程、Ray Actor、普通对象和 ActorHandle 拓扑；
- HYBRID 与 STANDALONE 初始化；
- Hybrid Engine 与 Colocated deployment 概念区分；
- one-step async 和 fully async 多种运行模式；
- partial rollout、陈旧度和 off-policy 控制；
- Checkpoint Engine 控制面、数据面及 `ServerAdapter`；
- STANDALONE 推理空泡检测；
- 强制回收和请求续推；
- 动态创建 vLLM replica/HTTP server；
- ResourcePool/PG 与 node ID/GPU ID 直接放置的差异。

归档入口：`multi_task_scheduler/references/archive/verl-v0.8.0/README.md`。

### 4.2 当前参考资料

| 文档 | 内容 | 基线/状态 |
|---|---|---|
| `00-project-alignment.md` | 项目背景和需求对齐 | 持续维护 |
| `01-redesign-scope.md` | 重设计范围和边界 | 持续维护 |
| `02-group-scheduler-protocol.md` | GroupScheduler 控制协议 | 设计文档 |
| `03-hybrid-standalone-component-topology.md` | 两种主部署模式组件拓扑 | 待继续按 v0.9 校对 |
| `04-hybrid-initialization-process.md` | HYBRID 初始化和动态创建 | 主要来自前期研究 |
| `05-standalone-initialization-process.md` | STANDALONE 初始化和动态创建 | 主要来自前期研究 |
| `06-standalone-inference-resource-bubble-detection.md` | 异步模式空泡和长尾判断 | 已多轮修正 |
| `07-async-parameter-sync-checkpoint-engine.md` | 异步参数同步和有效 replica | 已形成核心思路 |
| `08-forced-reclaim-request-continuation.md` | 强制回收和 partial rollout | 待结合 v0.9 原生能力更新 |
| `09-ray-worker-group-actor-binding-mechanism.md` | RayWorkerGroup 与 Actor 绑定 | v0.8.0 基线 |
| `10-training-side-worker-group-business-worker-association.md` | 训练侧 WorkerDict/业务 Worker | v0.8.0 基线 |
| `11-training-worker-dict-ray-actor-creation-call-chain.md` | v0.9 WorkerDict Actor 完整创建链 | 新增，待评审 |
| `12-python-class-instance-call-and-rayclasswithinitargs.md` | Python 调用语义和 CIA | 新增，待评审 |
| `14-verl-v0.8-v0.9-hybrid-standalone-differences.md` | v0.8.0→v0.9.0 两种主部署模式的架构、组件和流程差异 | 新增，待评审 |
| `15-verl-v0.9-hybrid-colocate-async-partial-rollout-interrupt-resume.md` | HYBRID + `colocate_async` 的中断、sleep、权重同步、续推和状态归属全链 | 新增，待评审 |
| `16-verl-v0.9-dynamic-replica-view-and-reference-audit.md` | HYBRID/STANDALONE 动态 replica 的引用、视图、发布事务和 GAP | 新增，待评审 |
| `17-verl-v0.9-dynamic-replica-scaling-timing-and-mutual-exclusion.md` | STANDALONE Fully Async、HYBRID partial、HYBRID 同步三种主链的扩缩容时机、CE snapshot、LB drain 和 lifecycle/native gate | 已按场景全量重构，待评审 |

正式工作文档：`multi_task_scheduler/多RL任务资源共享调度对接VERL - 架构及组件.md` 与
`multi_task_scheduler/[WIP] 多RL任务资源共享调度对接VERL - 关键流程.md`。当前文档仍在用户评审中，更新前先检查工作区差异。

## 5. 最近完成的 v0.9.0 代码结论

### 5.1 共卡 partial rollout

`PPOTrainerColocateAsync` 的当前机制已经确认：

- 使用 `FullyAsyncLLMServerClient`；
- ReplayBuffer 只向训练侧返回完整 trajectory；
- 收集够完整样本后，Trainer abort 其他未完成请求并 sleep rollout；
- vLLM 返回当前 token 前缀和 `stop_reason=aborted`；
- 客户端协程保存 token、logprob、剩余预算及 min/max global steps；
- 训练和权重同步完成后，客户端使用原 prompt 加已生成前缀重新 prefill 并续推；
- KV cache 不需要跨 sleep 保留；
- 一条 trajectory 可以跨多个模型版本；
- partial rollout 不表示使用半条 trajectory 更新 PPO。

进一步代码校验已经确认：

- 原生 hook 对 `CheckpointEngineManager.replicas` 全量 abort，不支持只指定一个待回收 replica；
- partial token、logprob、剩余预算和版本范围保存在 `AgentLoopWorkerTQ` 进程内的 client coroutine，不在
  TransferQueue、`vLLMReplica` 或 `CheckpointEngineManager`；
- 正常 resume 不从 TransferQueue 读取中断 prompt；同一个 `FullyAsyncLLMServerClient.generate()` 活协程帧直接读取
  原始 `prompt_ids` 和累计 `final_output.token_ids`，拼成下一次 backend 输入；
- `FullyAsyncLLMServerClient` 普通对象本身没有 per-request 状态表；从 `background_tasks`、`_run_prompt` task、session
  task、`AgentLoop.run()` 到 client generate coroutine frame 的 Python 引用链负责保活 partial 状态；
- TQ 只保存 prompt 状态、恢复所需 prompt 数据和最终完整 trajectory；进程故障后的 reissue 会从原 prompt 重生成，
  不恢复 partial prefix；
- resume 由 client while-loop 用同一 logical request ID 重新 acquire，并默认以新 backend UUID 提交
  `prompt + prefix`；`resume_generation_replicas()` 只解除服务端 pause；
- 当前共卡 hook 不在 abort/sleep 时从 LB 摘除 server，不能直接当成多任务 donor 回收协议；
- vLLM abort 返回值当前未由 `CheckpointEngineManager` 向 Trainer 传播，multi-node replica 还存在非 head server
  abort error 被吞掉的实测校验项。

该能力可以作为 GroupScheduler 强制回收的基础，但跨 replica 迁移、持久化和故障恢复仍未由原生机制完整覆盖。
完整证据链见 `15-verl-v0.9-hybrid-colocate-async-partial-rollout-interrupt-resume.md`。

文档 `15` 的 Mermaid 图已改为兼容性更高的基础语法：复杂并发图拆分为控制链和 partial 数据链，并新增 coroutine
状态归属图；本轮同时补齐 `TokenOutput`、TransferQueue prompt record 和 checkpoint reissue 的数据结构与取数流程。
使用 `@mermaid-js/mermaid-cli 11.16.0` 实际生成 SVG/PNG，7/7 个图均渲染成功并完成视觉检查。

### 5.2 STANDALONE `CheckpointEngineWorker`

已确认：

- `RolloutReplica.init_standalone()` 创建独立 ResourcePool 和 `RayWorkerGroup`；
- `get_ray_class_with_init_args()` 返回包装 `ActorClass(CheckpointEngineWorker)` 的
  `RayClassWithInitArgs`；
- `RayWorkerGroup._create_worker()` 调用该包装实例；
- `RayClassWithInitArgs.__call__()` 最终执行
  `ActorClass(CheckpointEngineWorker).options(...).remote(...)`；
- `.remote()` 真正创建 `CheckpointEngineWorker` Ray Actor 并返回 ActorHandle；
- `RayWorkerGroup` 是 controller 普通对象，其 `_workers` 保存 ActorHandle 列表。

### 5.3 训练侧 `WorkerDict`

已确认：

- Trainer 最初对 `ActorRolloutRefWorker`、`DetachActorWorker` 或 critic `TrainingWorker` 调用
  `ray.remote()`，但此时只得到业务 ActorClass 描述；
- `create_colocated_worker_cls()` 将业务 ActorClass 和构造参数收集进闭包；
- 函数动态定义 `WorkerDict`，并在其 `__init__()` 中解包业务 ActorClass 后普通构造业务对象；
- `ray.remote(WorkerDict)` 生成外层 ActorClass；
- 传给训练侧 `RayWorkerGroup` 的是包装 `ActorClass(WorkerDict)` 的第二层
  `RayClassWithInitArgs`；
- 最终通过 `.remote()` 创建的实际 Ray Actor 是 `WorkerDict`；
- `ActorRolloutRefWorker`、`DetachActorWorker` 和 critic `TrainingWorker` 是
  `WorkerDict` Actor 进程内普通对象；
- `spawn()` 只生成引用同一组 ActorHandle 的角色级 `RayWorkerGroup` 代理，不创建新 Actor。

### 5.4 `_bind_workers_method_to_parent()`

已确认该函数实现：

- 只扫描带 verl `@register`/`MAGIC_ATTR` 的业务接口；
- 在 `WorkerDict` 上生成委托到 `self.worker_dict[key]` 的代理方法；
- 默认使用 `<role>_<method>` 前缀避免 actor/critic 同名接口冲突；
- 复制 dispatch、collect、blocking 等调用元数据；
- 在 `ray.remote(WorkerDict)` 前完成绑定，使 ActorHandle 能暴露对应远端方法；
- 配合 `spawn()` 向 Trainer 提供无前缀的角色级透明调用视图。

### 5.5 Python 类调用和实例调用

已确认：

```text
RayClassWithInitArgs(...)
  → 调用类对象
  → type.__call__
  → __new__ + __init__
  → 返回 RayClassWithInitArgs 实例

ray_cls_with_init(...)
  → 调用上述实例
  → RayClassWithInitArgs.__call__
  → ActorClass.remote()
  → 返回 ActorHandle
```

该结论用于避免把 `replica.py:234` 的类构造误判为实例 `__call__()`。

### 5.6 v0.8.0 到 v0.9.0 的部署和 Trainer 架构差异

已基于精确 tag `v0.8.0@7aed6b23` 与 `v0.9.0@483b8a00` 完成对照，确认：

- `RolloutMode.HYBRID/STANDALONE` 的定义和 `RolloutReplica` 基本初始化路径没有本质变化；
- v0.9.0 把 `main_ppo_sync.py` 重构为 V1 `PPOTrainer` 抽象基类，并新增 `PPOTrainerSync`、
  `PPOTrainerColocateAsync`、`PPOTrainerSeparateAsync`；
- v0.9.0 默认 `trainer.use_v1=true`，但 `use_v1=false` 仍可进入
  `main_ppo_v0.TaskRunner + RayPPOTrainer` 兼容链；当前多任务分析以 V1 为目标主链；
- `trainer_mode=colocate_async` 仍使用 HYBRID rollout 资源组织，不等于 `RolloutMode.COLOCATED`；
- v0.9.0 共卡模式新增基于 abort/sleep/resume 和 `FullyAsyncLLMServerClient` 的 partial rollout；
- V1 separate 在一个 TaskRunnerV1 进程中持有 hybrid/standalone 两套 `LLMServerManager` 和两套
  `CheckpointEngineManager`，不再采用 V0.8 experimental 的 Trainer/Rollouter/MessageQueue 三控制 Actor 主链；
- V1 separate 的常规 HYBRID 再激活判断 `should_switch_to_rollout()` 当前固定返回 `False`，完整策略仍是 GAP；
- v0.9.0 experimental 新增 `DynamicResourceController`，但它只实现单任务内部 HYBRID/STANDALONE 切换，
  不能替代跨任务 GroupScheduler；
- v0.9.0 已提供 LB add/remove、partial retry、CE replica add/remove 等可复用执行原语，但动态 borrower
  Worker/replica 的跨任务物理创建仍未解决。

详细证据和流程见 `14-verl-v0.8-v0.9-hybrid-standalone-differences.md`。

### 5.7 HYBRID / STANDALONE 动态 replica 视图与引用

已完成独立评审文档 `16-verl-v0.9-dynamic-replica-view-and-reference-audit.md`，确认：

- 直接持有 `RolloutReplica` 对象的核心原生字段是 `LLMServerManager.rollout_replicas` 和
  `CheckpointEngineManager.replicas`；LB、ServerAdapter、ResourcePool 和 WorkerGroup 持有的是派生引用；
- V1 同进程路径中 manager/CE 初始共享同一个 list，但 CE add 原地 extend、remove 重新绑定 list，这种 alias 会在
  remove 后失效，不能作为动态一致性协议；
- Fully Async 的 canonical manager 位于 Rollouter Actor，CE 位于 Trainer Actor 并持有 replica 序列化副本；
  原生 remove 依赖对象 identity，必须改用 stable replica ID；
- STANDALONE 非 naive 同步会遍历 `CE.replicas[*].workers`，所以 CE 的有效 replica 集合能改变同步接收端；最新第一版设计用
  replica-sync gate 保护唯一 `effective_replicas`，不再复制 native membership snapshot；
- HYBRID naive 同步直接调用 `actor_wg.update_weights()`，真实目标由训练 workers 内的 `ServerAdapter` 静态映射决定，
  只修改 `CE.replicas` 不能同步一个真正新建的 borrowed replica；
- base `LLMServerManager` 没有通用运行期 materialize/teardown API；experimental add/remove 只激活初始化时已经创建的
  HYBRID replicas，不等于跨任务动态创建；
- vLLM 的 STANDALONE `sleep()`/`wake_up()` 当前跳过执行，KV cache release/resume 也没有实际动作，不能据此形成
  可捐赠 HBM 空泡；
- 动态添加在 borrower 内部必须接入三处：manager canonical ownership/lifecycle registry、Checkpoint Engine/sync
  projection、LB routing projection；任务外还必须维护 GroupScheduler node/GPU lease。容量/监控是派生更新。
- borrowed overlay 下 Ray 原生 pool/PG 保持 donor 的原预约是预期差异，但 donor 不管理 borrowed replica，也不把
  任何 runtime handle 交给 borrower；排他性由 lease、fencing、发布顺序和回滚协议约束。
- 当前 WIP RFC `:129-130,142-143,171-184` 的 donor CheckpointEngineWorker endpoint 重绑定/临时使用方案与最新约束
  冲突，本轮只在独立评审文档中标记，尚未修改 RFC。

本轮只输出独立评审材料，没有修改正式 WIP RFC。

### 5.8 动态 replica 扩缩容时机与互斥

已按用户确认的三种运行模式全量重构独立评审文档
`17-verl-v0.9-dynamic-replica-scaling-timing-and-mutual-exclusion.md`：

- STANDALONE 主链改为 experimental `FullyAsyncTaskRunner -> FullyAsyncTrainer / FullyAsyncRollouter`，不再用
  `PPOTrainerSeparateAsync` 代替 Fully Async；该模式必须同时使用训练 step、参数版本周期和并行 rollout 三条时间轴；
- STANDALONE Fully Async 与 HYBRID + partial 已按相同结构重组：AS-IS 类图、AS-IS 运行流程、TO-BE scaling 互斥规则、
  ADD 时序、REMOVE 时序、数值例子和 AS-IS/TO-BE/GAP；
- 该场景以 `PPOTrainerColocateAsync` 的 dispatch、`ReplayBufferAsync.sample()`、abort/sleep、training、step-end
  sync/resume 为完整 step，并显式区分 `TaskRunnerV1`、`WorkerDict`、`AgentLoopWorkerTQ`、`vLLMHttpServer`、运行时 LB 等
  Ray Actor 与 Trainer/manager/CE/replica/client 等普通对象；
- HYBRID + 同步以 `PPOTrainerSync` 的自然 drain/sleep、training、step-end sync 为完整 step；
- 文档已删除 STANDALONE Fully Async 与 HYBRID + partial 的时机矩阵，并把矩阵中的 CREATE、target-only BOOTSTRAP、
  CE/lifecycle 投影、LB ROUTABLE COMMIT、LB REMOVE、CE/lifecycle REMOVE 和 DESTROY 约束迁移为分阶段规则与竞态例子；
- 文档已取消两章中独立的“关键线性化点”表述。ADD/REMOVE 时序后的编号流程直接说明 CE、LB 和 runtime 状态何时生效；
- 最新确认：BOOTSTRAPPING 只把 new replica 更新到当前 `V_serving`，不递增任务参数版本、不 reset 全局 staleness、
  不 abort 或重复传输已有 replica；verl 原生参数同步才发布 `V_next` 并更新全部 effective replica；
- 最新确认：STANDALONE + Fully Async 第一版只维护 CE 的唯一 `effective_replicas`，并使用一把 task-local
  replica-sync gate 端到端串行 target-only bootstrap、CE ADD/REMOVE、LB `ROUTABLE` commit/rollback 和 verl native
  `update_weights()`。该模式不维护 desired membership、独立 membership gate 或 `native_sync_snapshot`；
- STANDALONE native sync 在整个 `CheckpointEngineManager.update_weights()` 调用期间持有 replica-sync gate，并直接多次读取同一个
  `effective_replicas`。所有 ADD/REMOVE 与回滚必须经过同一 gate，所以同步期间集合不会变化；
- STANDALONE native sync 先获得 gate 时，新 replica 等同步发布最新 serving version 后只 bootstrap 该版本；ADD 先获得 gate 时，
  ADD 在释放 gate 前完成 bootstrap、`effective_replicas` ADD 和 LB `ROUTABLE`，后续 native sync 直接遍历包含 new replica 的集合；
- HYBRID 继续使用 weight-version gate、membership gate 和 H4 immutable snapshot；STANDALONE 的单集合特化不自动推广到 HYBRID；
- trainer live weights 可能领先 serving replica，因此 bootstrap 需要可 pin 的不可变 `PublishedWeightSnapshot(V)`；HYBRID partial
  在 H2 更新 actor 后同样不能把 live weights 冒充旧 `V_serving`；
- Fully Async 的运行时主登记表位于 `FullyAsyncRollouter` Actor，CE 位于 `FullyAsyncTrainer` Actor 并持有 replica
  序列化副本，动态伸缩必须做跨 Actor 提交；LB 返回 `ROUTABLE` 后，新 acquire 才能看到 server；
- Fully Async 强制回收只有在 `async_training.partial_rollout=True` 时才能复用 prefix retry；关闭时按自然 drain 处理；
- 最新确认：HYBRID partial borrowed replica 可以跨过 `on_sample_end()`、training、H4 native sync 并保留到下一 step；H1 只对
  lifecycle snapshot abort/sleep，不执行 CE/LB REMOVE、runtime DESTROY 或 lease release；
- HYBRID partial 可在 LB 摘流后复用 target/full abort 和 client coroutine 中的 prefix 续推；partial prefix 不在 TransferQueue；
- HYBRID partial 的 AS-IS 陈旧度门禁已按代码校正：`ReplayBufferAsync` 使用
  `prompt_age = g_train - g_prompt + 1` 做 drop/wait，`TokenOutput.min/max_global_steps` 记录实际生成所跨 rollout 权重版本，
  但当前 sampler 不直接以该版本跨度做准入判断；
- HYBRID 同步没有 transparent partial retry，deadline 小于剩余请求时间时必须 defer/reject，不能强杀后声称样本可续推；
- HYBRID naive CE 只调用固定 `actor_wg.update_weights()`。HYBRID partial 的 retained B 必须进入 H4 immutable snapshot，因此需要
  target-only external bootstrap，以及 fixed/borrowed subsets 的联合发布；
- CREATE、lease reserve 和 LB begin-drain 不需要持有全局大锁。STANDALONE ADD 从版本 pin 到 `ROUTABLE/rollback` 使用
  replica-sync gate；HYBRID ADD 使用 weight-version gate。HYBRID partial 另需 task-local rollout-admission gate：ADD 激活只能在 rollout 准入打开时执行；
  `on_sample_end()` 等正在激活的 ADD 提交或回滚后关闭准入，H1–H4 只允许 hidden CREATE，H5 发布并 resume 后重新打开准入；
- `CheckpointEngineManager.update_weights()` 当前从 abort、worker flatten、KV lifecycle 到 resume 多次读取可变
  `self.replicas`。STANDALONE 第一版由外层 replica-sync gate 覆盖整个调用，因此不需要复制 immutable tuple；HYBRID 仍需冻结
  immutable snapshot 并用 sync/lifecycle refcount 保护销毁；
- 当前 `GlobalRequestLoadBalancer.remove_servers()` 会立即删除 inflight counter，后到的 fire-and-forget release 被忽略；
  安全回收需要 `begin_drain/finish_remove` 两阶段协议，同时验证 LB inflight 和 backend queued/running；
- `TaskRunnerV1` 和 `FullyAsyncTaskRunner` 都以同步长方法占用各自 Ray Actor；运行期间新增普通 actor method 不可达；
  子仓 TaskRunner 需要 control concurrency group，控制入口先校验 task session、operation ID 和 lease epoch。HYBRID 在
  rollout-admission、lifecycle 和 native-sync hook 处分别保护准入状态或冻结 snapshot；Fully Async 通过既有
  Trainer/Rollouter ActorHandle 做跨 Actor 提交；
- borrower `on_sample_end()` 不是 lease deadline。B 可以 sleep 后跨 step 保留；donor training 前需要证明 B 无 active compute、
  残余 HBM 可共存，或由 GS 独立 recall；`PPOTrainerColocateAsync` 的显式 recall 可以复用 partial retry；
- STANDALONE lease 可以跨 donor training 阶段，但 donor 必须先获得 replica-sync gate，并把 dormant native replica 从 donor
  `effective_replicas` 删除；资源归还后，donor dormant replica 同样在 gate 内先 target-only bootstrap 到 donor 当前 serving
  version，再恢复 CE effective/LB `ROUTABLE`；
- GroupScheduler 不改变原生 sync hook 或发布版本。bootstrap gate 可以延迟 hook 数据面；HYBRID partial H4 snapshot 必须包含仍
  effective 的 borrowed replica。GroupScheduler 只提交 scale command/lease/recall，不直接调用 CE，也不参与权重版本选择；
- validation 默认冻结 scaling membership；checkpoint save 不直接遍历 rollout replica，不需要与所有 prepare/drain 操作
  粗粒度互斥，但 effective/LB commit 仍受 validation topology gate 和 task session fencing 约束。
- 两章中的 AS-IS 图只使用真实类名；TO-BE 类统一标记为 Proposed。每张 ADD/REMOVE 时序图后都新增了具有明确主语、宾语和动作的
  编号流程说明；自定义术语在首次使用处补充了定义。
- 文档中的 11 个 Mermaid 图已使用本地 Mermaid CLI 11.16.0 全量实际渲染成功；本轮只读取 verl 代码，没有运行或修改 verl。

STANDALONE 第一版单 gate 模型的成立约束已经补充到文档：`FullyAsyncTrainer` 必须成为 `effective_replicas` 的唯一写入者；所有
CE mutation 和 native sync 必须使用同一 gate；native sync 必须端到端持锁；跨 Actor LB RPC 不得同步回调持锁的 Trainer 方法；
失败路径必须使用 operation ID、lease epoch、超时和幂等回滚。下一步先采集 gate wait/native sync/bootstrap 等指标；只有压测证明
单 gate 成为瓶颈后，才考虑批量事务或引入独立 epoch snapshot、desired membership、membership gate 和 sync pin。

本轮仍未修改正式架构/流程文档。rollout-admission gate 是本次为避免 H4/ADD 循环等待提出的 TO-BE 方案，等待用户评审，不属于
已经确认的 verl AS-IS 能力。

## 6. 当前未决问题

### 6.1 动态 borrower rollout 的 Actor 创建

需要基于 v0.9.0 继续明确：

- 不复用 donor ResourcePool/PG/bundle/worker/adapter 时，STANDALONE 如何创建 borrower-owned sync receiver；
- receiver 是否仍应是 `CheckpointEngineWorker` Ray Actor，若是，如何在不申请第二份 Ray GPU 的情况下创建；
- 是否需要直接使用 NodeAffinity、显式 CUDA visible devices 和定制 Ray options；
- 新 Actor 如何与 donor sleep 的 vLLM 进程共享同一物理 GPU而不被 Ray 原生资源调度冲突；
- borrower manager 如何持有新 endpoint handle，并如何组装 `RolloutReplica.workers` 与 Trainer CE 的 `effective_replicas`。

新增约束：HYBRID 不能把 donor worker handles 追加到 borrower CE list，因为 naive 同步目标由 borrower `actor_wg`
内既有 `ServerAdapter` 决定；STANDALONE 也只有在 `.workers` 指向 borrower-owned CE/adapter endpoints 时才能进入原生
同步拓扑。两种模式都禁止 donor endpoint rebind。

### 6.2 动态 vLLM replica 和 HTTP server

需要明确完整创建和回收顺序：

```text
选择 donor GPU
→ donor server 移出 LB
→ abort/drain
→ donor sleep
→ GroupScheduler reserve node/GPU lease
→ borrower manager 创建并登记自己的 vLLMHttpServer/backend
→ 创建或绑定 borrower-owned sync endpoint
→ 启动 vLLM WorkerProc
→ STANDALONE 获取 replica-sync gate；HYBRID 获取 weight-version gate
→ pin 当前 PublishedWeightSnapshot(V_serving)
→ 只向 new replica 执行 target-only bootstrap
→ 在 gate 内更新 borrower CE effective membership
→ 最后提交 borrower head server 为 LB ROUTABLE
→ 释放 gate；STANDALONE 后续 native sync 直接遍历包含 new replica 的 effective_replicas
→ HYBRID 后续 native sync snapshot 必须包含 new replica
→ GroupScheduler commit lease ACTIVE
```

还需要定义任何一步失败后的回滚顺序。

### 6.3 v0.9 partial rollout 与强制回收的复用边界

原生行为已经确认：

- Trainer hook 全量 abort manager 中的 replicas；
- client coroutine 持有可重发的 token prefix，LB server 被移除时能够重选，但原生 hook 不执行摘流或跨任务迁移；
- partial prefix 没有持久化到 TransferQueue；
- logical request ID 用于 LB sticky，backend vLLM ID 默认每个 attempt 新建；
- abort 结果、TQ uid 与 backend request ID 之间没有供调度器使用的闭环映射。

仍需设计和评审：

- per-replica targeting 与 LB 两阶段摘流；
- 是否以及如何持久化 partial token、logprob 和版本状态；
- borrower 版本门禁、请求唯一所有权和幂等重试；
- 如何避免请求重复完成、丢失或被两个实例同时续推；
- multi-node replica 的 abort/pause 广播和错误处理。

### 6.4 参数同步有效集合

文档 `17` 已明确时间和互斥原则：

- bootstrap 与原生 sync 触发相互独立。STANDALONE 使用 replica-sync gate 同时保护唯一 `effective_replicas` 和整个
  `update_weights()`；HYBRID 使用 weight-version/membership gate 保护 H4 snapshot 和传权；
- STANDALONE Fully Async 在 new replica runtime ready 后立即请求只同步当前 `V_serving`，不等待下一原生同步周期，也不更新已有
  replica；HYBRID partial 在 rollout 准入关闭时只保留 hidden runtime，H5 后同步最新 `V_serving`；
- STANDALONE 不维护 desired membership 或 native snapshot；等待 gate 的 ADD/REMOVE 只保存在 TaskRunner operation journal；
- STANDALONE replica 可以先进入 DRAINING。REMOVE 在请求排空后获取 replica-sync gate，从 `effective_replicas` 删除目标并取得 CE
  remove receipt，然后才能销毁 runtime；
- HYBRID desired membership 可以在 sync 期间记录，但当前 H4 epoch 使用 immutable effective snapshot；已被 snapshot pin 的
  replica 可以先进入 DRAINING，但不能销毁；
- 新增实例 bootstrap、CE effective commit 或 LB commit 失败时保持 hidden/failed，回滚 effective membership 并释放 gate；
- HYBRID partial borrowed replica 可跨 step 保留；`on_sample_end()` 先关闭 rollout 准入，再冻结 lifecycle snapshot 并
  abort/sleep。H4 native snapshot 包含 fixed native 与当前 effective borrowed replica；H5 发布并 resume 后重新打开准入；
- donor native replica 归还后也先 target-only bootstrap donor 当前 serving version，再恢复 CE effective/LB；
- GroupScheduler 的 ADD 命令可以启动 task-local bootstrap 事务，但 GroupScheduler 不直接执行 bootstrap/原生同步数据面，
  也不选择或发布权重版本；它只下发规模命令和物理 lease。

仍需明确：

- STANDALONE replica-sync gate owner、超时、幂等回滚、跨 Actor 禁止回调与 restart reconcile 的具体 API；
- HYBRID immutable native snapshot/pin、weight-version gate owner、rollout-admission token 的具体 API 和异常恢复；
- `PublishedWeightSnapshot(V)` 的生成、保留、pin/unpin 与 target-only replay 接口；
- HYBRID lifecycle snapshot、native effective snapshot、sync/lifecycle pin 与 mixed-subset receipt 的具体 API；
- 跨任务模型、tokenizer、backend 和并行配置不兼容时如何在 lease 前拒绝共享；
- bootstrap/native sync receipt 的数据结构、版本校验和 LB publish 超时补偿。

还需分别解决：

- HYBRID partial 真正新建 replica 的 dynamic external receiver，以及 native fixed/borrowed subsets 联合发布；
- STANDALONE borrowed replica 的 borrower-owned target receiver endpoint 创建；
- STANDALONE native sync 如何在外层 replica-sync gate 内复用原生 `CheckpointEngineManager.update_weights()`；
- HYBRID 一次 native sync 如何从 worker 展平到 finalize 全程使用同一个不可变 effective snapshot。

### 6.5 三种视图的一致性协议

尚需形成状态机，覆盖：

```text
ACTIVE_DONOR
→ DRAINING
→ SLEEPING
→ DONATED
→ BORROWER_CREATING
→ BORROWER_BOOTSTRAPPING
→ BORROWER_CE_EFFECTIVE
→ BORROWER_ACTIVE
↔ BORROWER_DORMANT_RETAINED（可跨 `on_sample_end()` / step，再次 update/wake 后回到 ACTIVE）
→ RECLAIMING
→ DONOR_WAKING
→ ACTIVE_DONOR
```

每个状态需要定义原生 ResourcePool 视图、LB 视图和 GroupScheduler 视图，并规定失败回滚目标。

### 6.6 文档版本迁移

- `09`、`10` 仍是 v0.8.0 基线，应决定是归档还是按 v0.9.0 全量重写。
- `03`、`04`、`05`、`06`、`07`、`08` 需要逐步按 v0.9.0 重新校对代码和行号。
- v0.8.0→v0.9.0 的总体差异已落入独立文档 `14`，共卡 partial rollout 全链已落入独立文档 `15`，但尚未
  回写上述主文档。
- 正式 RFC 应在上述关键问题评审后更新，避免把未确认设计写成定论。

## 7. 下一步建议顺序

推荐按以下顺序继续：

1. 联合评审文档 `16` 的 M/C/L/G 视图边界，以及文档 `17` 的 rollout-admission、lifecycle、weight-version 和 native snapshot
   gate，先确认原子性和时间语义。
2. 设计 HYBRID partial 的 rollout-admission token、retained-borrowed lifecycle snapshot、H4 native + borrowed immutable
   snapshot、pin 和 mixed-subset `NativeSyncReceipt` API。
3. 设计 `PublishedWeightSnapshot(V)`、target-only `bootstrap_replica()`、weight-version gate、CE immutable effective
   snapshot/pin 与 LB `begin_drain/finish_remove/commit_routable` 的最小 API，
   并覆盖 native-sync-first、bootstrap-first、超时和取消四类并发测试。
4. 选择 HYBRID borrowed replica 的动态权重 endpoint 方案，并证明 sleeping B 跨 donor training 保留时的残余 HBM、CUDA/NCCL
   状态和 active-use fencing；不能继续把 `CE.add_replicas()` 当作完整答案。
5. 设计并验证 STANDALONE 的真实 HBM release，以及 borrower CE/ServerAdapter target endpoint 创建方案。
6. 基于文档 `15` 的原生边界，设计 per-replica 摘流、请求唯一所有权和跨 replica continuation 协议。
7. 经用户评审后再更新 `07`、`08` 和正式架构/流程文档；当前不修改它们。

## 8. 进度维护规则

后续每完成一个实质阶段，更新：

- 当前代码基线；
- 新增或修改的文档；
- 已确认结论；
- 被推翻或修正的历史结论；
- 新增未决问题；
- 推荐下一步；
- Git 提交和推送状态。

长期分析原则和用户已确认的稳定约束写入根目录 `AGENTS.md`；易变化的实现进度只写入本文。
