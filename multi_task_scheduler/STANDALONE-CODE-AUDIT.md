# STANDALONE 全量代码校对

> 校对对象：[`STANDALONE.md`](STANDALONE.md)。
>
> verl 基线：`/Users/nyp/Documents/verl`，commit `7aed6b230776f963fa09509c10d9c3a767d1102c`。
>
> GroupScheduler 模拟器参考：`/Users/nyp/Documents/multi-rl-task-scheduler`，HEAD
> `7ac1396a60bc6d22b9f2e908f9440176781d3f3e`。校对时该仓库 worktree **有未提交修改**；本文 `[S]` 结论只引用经
> `git diff --quiet` 确认为与 HEAD 一致的
> `src/multi_rl_task_scheduler/{group_scheduler,models,interfaces,algorithms}.py` 和
> `group_scheduler/sleep_registry.py`。旧实验文件 `group_scheduler/group_scheduler.py:64-68` 的
> `@yr.instance`/`GroupScheduler` 声明已同时核对 HEAD 与当前 worktree，二者一致；该文件的其他改动不作为事实依据。
>
> 校对原则：代码是“现状”的唯一依据；设计项没有实现代码时，必须标成目标扩展，并给出可复用代码锚点和新增点，不能写成
> verl 或模拟器已有能力。

## 1. 结论和标记

本文使用四类标记：

| 标记 | 含义 | 可否称为“已有实现” |
|---|---|---|
| `[V]` | verl v0.8.0 已有代码事实 | 可以 |
| `[S]` | GroupScheduler 模拟器已有抽象/行为 | 只能称为模拟器能力 |
| `[D]` | 本项目目标设计，当前三个代码仓均无实现 | 不可以 |
| `[M]` | 可复用已有代码，但还必须新增语义 | 只能明确写成“扩展后” |

全量校对后的总判断：方案方向成立，但原文把若干目标态画成了运行现状。必须纠正的关键点如下。

1. `[D]` `GroupScheduler` 的生产形态“Ray Actor 单例”、TaskRunner 双向 ActorHandle、LB 直报 GS 均没有代码；主模拟器中的
   `GroupScheduler` 是普通内存对象（`src/multi_rl_task_scheduler/group_scheduler.py:18-45`），另一个旧实验实现使用
   `@yr.instance`（`group_scheduler/group_scheduler.py:64-68`；该声明在 HEAD 与当前 worktree 一致），都不是 Ray Actor。
2. `[V]` FullyAsync 的原生对象归属基本正确：TaskRunner、Trainer、Rollouter、MessageQueue、LB、AgentLoopWorker 都是
   Ray Actor；manager、client、replica descriptor 是 Actor 进程内普通 Python 对象。
3. `[V]` 原生 STANDALONE 确实为每个 replica 新建 `ResourcePoolManager → RayResourcePool → PG bundles →
   CheckpointEngineWorker actors`，再从 worker actor 获取 node/GPU 并用硬 NodeAffinity 创建 Server Actor。
4. `[V]` Server Actor 创建时没有申请 `num_gpus`，实际设备可见性来自 worker actor 的 accelerator ID 和显式
   `cuda_visible_devices`；但它仍依赖 worker/PG 提供稳定 placement anchor。
5. `[V]` 原生 LB 支持运行时 `add_servers/remove_servers`，但 remove 是立即删除；没有 READY/DRAINING/SYNCING 状态、route
   token、routing epoch、policy version 或摘流确认。
6. `[V]` 原生 CE 遍历显式 `replicas` 列表，已有 `add_replicas/remove_replicas`，动态列表方向可复用；但没有 membership
   lock、snapshot epoch、安装版本 ACK、LB fence 或事务回滚。
7. `[V]` 原生 NCCL 在 world size 改变时若 `rebuild_group=false` 会断言旧 rank/world size 不变；动态 membership 使用 NCCL
   必须令 `rebuild_group=true`。NIXL 每次 `finalize()` 都清理链路状态，拓扑也会按当前 worker 列表重建。
8. `[V]` verl v0.8.0 的 STANDALONE `vLLMHttpServer.sleep()` 和 `wake_up()` 明确跳过。因此“level-2 sleep 释放 HBM”是
   `[D]` 必须新增的核心能力，不能当作已有方法效果。
9. `[V]` FullyAsync 的 admission 上界是
   `int(required_samples * (staleness_threshold + 1) * trigger_parameter_sync_step)`；暂停条件是 MQ 满或
   `staleness_samples` 达到该上界。`active_tasks` 是 sample coroutine 集合，不是 per-replica inflight。
10. `[V]` partial rollout 的 client 重试只在 `config.async_training.partial_rollout=true` 且 stop reason 为 abort 时发生；没有
    abort reason，也没有资源抢占专用重试。vLLM abort 甚至可能返回空 token，所以“必然从 prefix 续推”不成立。
11. `[S]` 主模拟器当前执行路径只回收 idle instance；强制回收 busy replica 是 `[D]`，不能说模拟器已经覆盖。模拟器
    `SleepRegistry` 只证明抽象状态转换和整实例原子绑定，不证明真实 vLLM HBM 已释放。
12. `[D]` borrowed Server、borrowed CE receiver、显式 GPU 复用、三视图 lease/fence、同版本 LB 激活和异常恢复都没有产品
    代码，必须通过 PoC 验证 Ray/CUDA/vLLM 的共卡可行性。
13. `[M]` `FullyAsyncTaskRunner/Trainer/Rollouter` 都已经由 `@ray.remote` 装饰，导入符号是 ActorClass；文档中的 MultiTask 名称
    只能表示目标逻辑角色，不能直接当作普通 Python 可继承类。需要上游暴露未装饰 base，或保留原 Actor 并增加 manager 注入点和
    通用 membership RPC。
14. `[D]` 原 FullyAsync TaskRunner 默认单并发且同步 `run()` 阻塞整个训练期，无法在训练中同时接收 GS heartbeat/command RPC；
    MultiTask Actor 必须增加受控并发或改变监督方式。One-Step 虽有 `max_concurrency=100`，但 Trainer 当前只是 `run()` 局部变量，
    命令入口也无法访问。

校对版主图明确限定 `use_trainer_do_validate=false`。若为 true，FullyAsync 还会预创建 trainer-GPU HYBRID validation replicas 和
独立 hybrid CE manager，代码见 `experimental/fully_async_policy/fully_async_main.py:159-174`、
`fully_async_rollouter.py:177-226`、`fully_async_trainer.py:176-243`；它们不属于第一阶段 STANDALONE donor/borrower 集合。

## 2. 代码基线和源文件索引

### 2.1 verl 原生运行时

| 主题 | 代码依据 |
|---|---|
| Ray 初始化、创建 TaskRunner | `verl/trainer/main_ppo.py:52-102` |
| One-Step TaskRunner 和本地 Trainer | `verl/experimental/one_step_off_policy/main_ppo.py:34-107` |
| FullyAsync TaskRunner 组件表 | `verl/experimental/fully_async_policy/fully_async_main.py:35-115` |
| FullyAsync 创建 Trainer/Rollouter | 同文件 `:117-157` |
| FullyAsync 并发运行 Trainer/Rollouter | 同文件 `:176-209` |
| FullyAsync Trainer Actor、CE 所有权 | `verl/experimental/fully_async_policy/fully_async_trainer.py:52-53,135-174,249-254` |
| FullyAsync Rollouter Actor、manager 所有权 | `verl/experimental/fully_async_policy/fully_async_rollouter.py:392-515,797-812` |
| LLMServerManager 和 LB/client 创建 | `verl/workers/rollout/llm_server.py:223-364` |
| LB 数据结构和 add/remove | 同文件 `:43-143` |
| Client acquire/generate/release | 同文件 `:146-220` |
| STANDALONE replica/worker 创建 | `verl/workers/rollout/replica.py:189-239` |
| Ray PG bundle 和 worker GPU 记账 | `verl/single_controller/ray/base.py:112-160,182-213,536-680` |
| vLLM Server placement/启动 | `verl/workers/rollout/vllm_rollout/vllm_async_server.py:968-1054` |
| STANDALONE sleep/wake 现状 | 同文件 `:604-635,1056-1060` |
| vLLM abort 和 aborted output | 同文件 `:533-602,679-715,1062-1083` |
| CE worker/manager/同步流程 | `verl/checkpoint_engine/base.py:278-340,345-515` |
| Trainer 侧 CE sender | `verl/workers/engine_workers.py:618-629,666-700` |
| NCCL 动态 topology 约束 | `verl/checkpoint_engine/nccl_checkpoint_engine.py:103-170,200-225` |
| NIXL topology/finalize | `verl/checkpoint_engine/nixl_checkpoint_engine.py:238-369` |
| FullyAsync admission/空闲统计 | `verl/experimental/fully_async_policy/fully_async_rollouter.py:485-595,848-951,1077-1119` |
| FullyAsync 参数同步点 | `verl/experimental/fully_async_policy/fully_async_trainer.py:487-524` |
| partial rollout client | `verl/experimental/fully_async_policy/fully_async_rollouter.py:51-150` |
| MessageQueue Actor | `verl/experimental/fully_async_policy/message_queue.py:26-134,180-234` |
| AgentLoop manager/worker | `verl/experimental/agent_loop/agent_loop.py:393-455,1044-1119` |
| One-Step manager、CE 和迭代 | `verl/experimental/one_step_off_policy/ray_trainer.py:170-235,266-413`；`verl/experimental/separation/ray_trainer.py:106-131,645-650` |

### 2.2 GroupScheduler 模拟器

| 主题 | 代码依据 | 能证明什么 |
|---|---|---|
| GS 普通对象和内存账本 | `src/multi_rl_task_scheduler/group_scheduler.py:18-45` | 调度器对象、worker/task/state 账本；不能证明 Ray 单例 |
| task 注册/上报/注销 | 同文件 `:55-83` | 抽象协议存在；注册参数是 task-local `InferScheduler`，不是 TaskRunner handle |
| state version 防旧报告 | 同文件 `:85-121` | 上报单调版本语义可复用 |
| assign/reclaim 下发 | 同文件 `:123-157` | GS 调普通 `InferScheduler.assign/reclaim` 的抽象调用 |
| 只回收 idle instance | 同文件 `:212-257` | 主执行路径不支持 busy 强制回收 |
| worker/node/GPU 与 task state | `src/multi_rl_task_scheduler/models.py:7-58` | worker placement、idle/busy/current 等抽象状态 |
| GS/InferScheduler 接口 | `src/multi_rl_task_scheduler/interfaces.py:9-30` | 调度与任务局部执行器边界 |
| 调度收益和整实例卡数 | `src/multi_rl_task_scheduler/algorithms.py:11-46,97-158,161-230` | 策略模型；不等于真实 verl 生命周期 |
| ACTIVE/SLEEPING 原子 registry | `group_scheduler/sleep_registry.py:12-62,93-126,132-258` | 整实例绑定/睡眠/唤醒的模拟状态机 |

## 3. 图 1：总体组件类图逐节点校对

| 图中节点 | 校对 | 代码或扩展锚点 |
|---|---|---|
| `GroupScheduler <<Ray Actor>>` | `[D]` | 模拟器只有普通类：`src/multi_rl_task_scheduler/group_scheduler.py:18-45`。生产 Ray Actor、命名/获取单例均需新增 |
| `MultiTaskFullyAsyncTaskRunner` | `[M]` | 新建/替换原 `FullyAsyncTaskRunner` ActorClass；原编排和 `components`：`fully_async_main.py:35-115`，GS handle、命令入口需新增；不能假定普通继承 |
| `MultiTaskFullyAsyncTrainer` | `[M]` | 原 `FullyAsyncTrainer` 在 `fully_async_trainer.py:52-53` 已装饰为 ActorClass，不能把目标名理解为直接继承；需暴露 base 或向原 Actor 增加动态 CE membership RPC |
| `MultiTaskFullyAsyncRollouter` | `[M]` | 原 `FullyAsyncRollouter` 在 `fully_async_rollouter.py:392-393` 已装饰为 ActorClass；需暴露 base，或向原 Actor 增加 manager 注入/borrowed replica RPC |
| `MultiTaskCheckpointEngineManager` | `[M]` | 可扩展 `CheckpointEngineManager`：`checkpoint_engine/base.py:345-515`；lock/epoch/version ACK/fence 均需新增 |
| `MultiTaskLLMServerManager` | `[M]` | 可扩展 `FullyAsyncLLMServerManager`：`fully_async_rollouter.py:153-343`；该类当前只激活预注册 hybrid replica，不会创建 borrowed STANDALONE |
| `EffectiveReplica` | `[D]` | 原 CE 直接接收 `list[RolloutReplica]`：`checkpoint_engine/base.py:374-385`；descriptor/schema 需新增 |
| `NativeRolloutReplica` | `[V]` | 对应 `RolloutReplica`/`vLLMReplica`：`replica.py:70-129`、`vllm_async_server.py:952-1054`；“Native”仅是设计分类名 |
| `BorrowedRolloutReplica` | `[D]` | 无类；扩展点是 `RolloutReplica` 的 `workers/servers` 字段和 `launch_servers()`：`replica.py:122-129,241-244` |
| `CheckpointEngineWorker` | `[V]` | Ray remote wrapper 在 `replica.py:228-239`；实现类在 `checkpoint_engine/base.py:278-340` |
| `BorrowedCheckpointEngineWorker` | `[D]` | 无类；可复用 `CheckpointEngineWorker` 的 receive/apply 路径，但需新增无 Ray GPU 申请的受赠创建入口 |
| `MultiTaskGlobalRequestLoadBalancer` | `[M]` | 原 LB Actor：`llm_server.py:43-143`；状态机、GS handle、版本过滤、route token 需新增 |
| `AgentLoopManager` | `[V]` | 普通对象，创建 worker actors：`agent_loop.py:1044-1099` |
| `AgentLoopWorker` | `[V]` | Python 类由 manager 包装成 Ray Actor：`agent_loop.py:393-413,1068-1099` |
| `MessageQueue` | `[V]` | Ray Actor：`experimental/fully_async_policy/message_queue.py:26-134` |
| `MessageQueueClient` | `[V]` | 普通对象副本，持有 MQ ActorHandle：同文件 `:180-234`；注入 Trainer/Rollouter：`fully_async_main.py:94-103` |
| `TrainingWorker` | `[V]` | Ray worker actor 内含 trainer-side CheckpointEngine sender：`workers/engine_workers.py:618-629,666-700` |
| `MultiTaskLLMServerClient` | `[M]` | 扩展 `LLMServerClient`/`FullyAsyncLLMServerClient`：`llm_server.py:146-220`、`fully_async_rollouter.py:51-150` |

### 3.1 图中引用边校对

| 引用/调用边 | 校对 | 代码依据或缺口 |
|---|---|---|
| TaskRunner → Trainer/Rollouter ActorHandle | `[V]` | `fully_async_main.py:77-87,117-157` |
| Trainer → Rollouter ActorHandle | `[V]` | `fully_async_trainer.py:158-174,249-254` |
| Trainer → CE manager | `[V]` | `fully_async_trainer.py:167-174` |
| Rollouter → LLMServerManager/AgentLoopManager | `[V]` | `fully_async_rollouter.py:797-812` |
| LLMServerManager → replica objects/LB handle | `[V]` | `llm_server.py:304-341` |
| AgentLoopManager → AgentLoopWorker handles | `[V]` | `agent_loop.py:1079-1099` |
| AgentLoopWorker → client → LB handle | `[V]` | `agent_loop.py:393-413`、`llm_server.py:153-177` |
| TaskRunner/Trainer/Rollouter → MessageQueueClient → MQ handle | `[V]` | `fully_async_main.py:94-103`、`message_queue.py:180-234` |
| Trainer/CE manager → TrainingWorker handles | `[V]` | `fully_async_trainer.py:167-174,334-375`、`checkpoint_engine/base.py:374-411` |
| GS ↔ TaskRunner handles | `[D]` | 三个仓中无实现；模拟器注册的是普通 `InferScheduler`：`interfaces.py:9-30` |
| LB → GS 上报 | `[D]` | 原 LB 无 GS 成员：`llm_server.py:60-66` |
| CE → EffectiveReplica → borrowed worker | `[D]` | 原 CE 只保存可变 `replicas` list：`checkpoint_engine/base.py:374-429` |

### 3.2 可实际落地的注入点

| 注入点 | 当前代码 | 校对结论 |
|---|---|---|
| FullyAsync CE manager | `fully_async_trainer.py:27,167-174` 直接 import/构造原类 | `[D]` 增加 manager class FQN/factory，或抽未装饰 base |
| FullyAsync LLM manager | `fully_async_rollouter.py:803-812` 直接构造 `FullyAsyncLLMServerManager` | `[D]` 增加 manager class FQN/factory |
| FullyAsync CE membership RPC | Trainer Actor 无 add/remove endpoint | `[D]` 增加通用 RPC，内部调用扩展 manager |
| FullyAsync borrowed creation RPC | Rollouter 只有 `add_replicas(resource_ids)`，`:1134-1138` | `[D]` 新增 materialize/dematerialize descriptor RPC |
| FullyAsync TaskRunner command 并发 | `fully_async_main.py:35-50,176-209`：默认单并发、同步 run 长期阻塞 | `[D]` 配 concurrency groups/max_concurrency 或让 run 非阻塞；事务需 lock |
| One-Step CE manager | `separation/ray_trainer.py:120-131` 已支持 `checkpoint_manager_class` | `[V]` 可直接复用 |
| One-Step LLM manager | `one_step_off_policy/ray_trainer.py:170-196` 硬编码原 manager | `[D]` 增加与 CE 一致的注入点 |
| One-Step TaskRunner state | `one_step_off_policy/main_ppo.py:34-35,90-107`：有并发但 Trainer 是局部变量 | `[D]` 保存 `self.trainer` 并保护跨线程状态 |

## 4. 表 1：One-Step 与 FullyAsync 归属校对

| 维度 | One-Step 代码事实 | FullyAsync 代码事实 | 设计变化 |
|---|---|---|---|
| TaskRunner | `[V]` `OneStepTaskRunner` Ray Actor，`main_ppo.py:34-35` | `[V]` `FullyAsyncTaskRunner` Ray Actor，`fully_async_main.py:35-36` | 两者的 MultiTask 子类均 `[D]` |
| Trainer | `[V]` 普通对象，TaskRunner 内构造并直接调用，`main_ppo.py:90-107` | `[V]` 独立 Ray Actor，`fully_async_main.py:138-156` | FullyAsync 的 CE membership 必须经 Trainer RPC |
| Rollouter | `[V]` 没有独立 Rollouter Actor；manager 位于 Trainer 内，`experimental/one_step_off_policy/ray_trainer.py:170-196` | `[V]` 独立 Actor，`fully_async_main.py:117-135` | 原文写成“Trainer 内 AgentLoop 对象”是正确概括，但不是名为 Rollouter 的对象 |
| LLM manager | `[V]` Trainer 普通成员，`:190-196` | `[V]` Rollouter 普通成员，`fully_async_rollouter.py:797-812` | MultiTask 扩展类 `[D]` |
| CE manager | `[V]` Trainer 普通成员，`separation/ray_trainer.py:120-131` | `[V]` Trainer Actor 普通成员，`fully_async_trainer.py:167-174` | effective membership 事务 `[D]` |
| 同步入口 | `[V]` `_fit_update_weights()`，`separation/ray_trainer.py:645-650` | `[V]` `_fit_update_weights()`，`fully_async_trainer.py:501-524` | GS 不触发同步是 `[D]` 约束，不是已有防护 |

## 5. 图 2：原生 STANDALONE 部署校对

图中的实体应按以下事实解释。

| 实体/边 | 校对结果 | 代码依据 |
|---|---|---|
| 每个 replica 独立 ResourcePool/PG | `[V]` | `replica.py:189-208` |
| 每个 GPU bundle 一个 CheckpointEngineWorker Actor | `[V]` | bundle 是 1 GPU：`ray/base.py:143-155`；worker 按 bundle 创建：`:568-579,621-680` |
| worker actor 只记账 0.5 GPU | `[V]` | STANDALONE 设置 `max_colocate_count=2`：`replica.py:202-206`；actor `num_gpus=1/max_colocate_count`：`ray/base.py:627-678`。PG 本身仍预留完整 1 GPU bundle |
| Server Actor 与 worker 在同一节点 | `[V]` | 读取 worker node/GPU：`vllm_async_server.py:976-989`；硬 NodeAffinity：`:1015-1023` |
| Server Actor 不声明 Ray GPU | `[V]` | `.options(...)` 未设置 `num_gpus`：`:1015-1023` |
| Server 使用 worker GPU IDs | `[V]` | `cuda_visible_devices` 从 worker accelerator IDs 拼接：`:976-998,1024-1033` |
| 一个 replica 多节点时每节点一个 Server Actor | `[V]` | `for node_rank in range(nnodes)`：`:991-1034`；首 Server 暴露 LB endpoint：`:1047-1054` |
| HTTP Server 和 backend processes | `[V]` | Server Actor 调 `launch_server.remote()`：`:1036-1045`；backend 在 Server Actor 内由 vLLM 启动，不能把每个 backend process 画成 Ray Actor |
| Trainer CE → rollout CE workers | `[V]` | CE 汇总 replica worker handles 并构造临时 WG：`checkpoint_engine/base.py:485-502` |
| “TCE”是单独 Actor | **不正确** | Trainer 侧 CE 是训练 worker 进程内对象：`engine_workers.py:618-629`，不是额外 Ray Actor。更新后的图应画在 Training Worker 内 |

## 6. 图 3：donor/borrower 共卡部署校对

整张图是 `[D]` 目标部署，不是当前运行图；其中可复用事实只有 placement discovery、显式 device visibility、LB 动态注册和 CE 显式
replica list。

| 设计断言 | 校对 | 可复用代码锚点 / 必须新增 |
|---|---|---|
| donor PG/worker 保持存活 | `[D]` 策略 | 原生 replica 持有 `resource_pool/workers`：`replica.py:122-129`；需新生命周期编排 |
| donor Server level-2 sleep | `[D]` | 原生 STANDALONE 明确 skip：`vllm_async_server.py:622-634`；需实现并验证真实 HBM 释放 |
| borrower Server 复用相同 node/GPU | `[M]` | placement 获取和显式 visible devices 可复用：`:976-1033`；lease 校验、跨任务 actor 创建需新增 |
| borrower actor 不申请 Ray GPU | `[D]` | Server actor 原生恰好不申请 GPU；CE worker 原生通过 WG 会申请 0.5 GPU，故 borrowed CE worker 必须有新的创建路径 |
| borrower CE receiver | `[D]` | `CheckpointEngineWorker` 可复用 receive/apply：`checkpoint_engine/base.py:278-325`；配置、server adapter 和设备绑定需新增 |
| GS 是唯一共卡账本 | `[D]` | 模拟器有 worker/task 账本：`src/multi_rl_task_scheduler/group_scheduler.py:27-45`；无 slot/lease epoch/Ray fencing |

不能从现有代码得出“绕过 Ray 资源视图一定安全”。至少要 PoC 验证：同一 GPU 上两个独立 Ray Actor/子进程的 CUDA context、vLLM
level-2 sleep、CE buffer 峰值、NCCL group 重建和进程销毁后显存回收。

## 7. 表 2：运行实体类型和部署位置校对

| 实体 | 正确类型 | 现状/设计 | 代码依据 |
|---|---|---|---|
| FullyAsyncTaskRunner | Ray Actor | `[V]` | `fully_async_main.py:35-36` |
| FullyAsyncTrainer | Ray Actor | `[V]` | `fully_async_trainer.py:52-53` |
| FullyAsyncRollouter | Ray Actor | `[V]` | `fully_async_rollouter.py:392-393` |
| MessageQueue | Ray Actor | `[V]` | `message_queue.py:26-27` |
| LLMServerManager/FullyAsyncLLMServerManager | 普通对象 | `[V]` 位于 Rollouter 或 One-Step Trainer 进程 | `llm_server.py:223-264`、`fully_async_rollouter.py:797-812` |
| CheckpointEngineManager | 普通对象 | `[V]` 位于 Trainer 进程 | `checkpoint_engine/base.py:345-385` |
| GlobalRequestLoadBalancer | Ray Actor | `[V]` CPU actor | `llm_server.py:43-66` |
| AgentLoopManager | 普通对象 | `[V]` | `agent_loop.py:1044-1077` |
| AgentLoopWorker | Ray Actor | `[V]` CPU node soft affinity | `agent_loop.py:1068-1099` |
| LLMServerClient | 普通对象副本 | `[V]` 在 AgentLoopWorker 中 | `agent_loop.py:393-413` |
| CheckpointEngineWorker | Ray Actor | `[V]` native PG bundle | `replica.py:217-239`、`ray/base.py:621-680` |
| vLLMHttpServer | Ray Actor | `[V]` hard node affinity、无 Ray GPU 声明 | `vllm_async_server.py:991-1033` |
| vLLM backend | Server Actor 启动的后端进程/engine | `[V]`，不是 Ray actor handle 列表 | `vllm_async_server.py:1036-1045` |
| GroupScheduler/MultiTask*/Borrowed* | 目标类型 | `[D]` | 当前子仓无 Python 实现；只能引用上述扩展锚点 |

## 8. 图 4：初始化时序逐边校对

| 时序边 | 校对 | 代码依据/修正 |
|---|---|---|
| TaskRunner 创建 Trainer | `[V]` | `fully_async_main.py:77-78,138-156` |
| TaskRunner 创建 Rollouter | `[V]` | 同文件 `:83-84,117-135` |
| Rollouter 创建 native replicas/LB | `[V]` | `fully_async_rollouter.py:797-812` → `llm_server.py:257-264,266-341` |
| native `init_standalone()` | `[V]` | `llm_server.py:297-325` → `replica.py:189-226` |
| Trainer `set_rollouter` 后构造 CE | `[V]` | `fully_async_main.py:86-87` → `fully_async_trainer.py:167-174,249-254` |
| 创建 MessageQueue 并注入两侧 | `[V]` | `fully_async_main.py:94-103` |
| 两侧 load checkpoint | `[V]` | 同文件 `:105-107` |
| initial `_fit_update_weights` | `[V]` | 同文件 `:109-110` |
| `CE -> LB: READY(V0)` | `[D]` | 原 CE 不引用 LB、不维护 READY/version。原生同步完成后 Server 可继续接流，但没有显式 READY 提交 |
| GS get/create/register | `[D]` | 无生产代码；模拟器 `register_task` 只接收 config/scheduler：`src/multi_rl_task_scheduler/group_scheduler.py:55-69` |
| “native descriptors 返回 TaskRunner” | `[D]` | 原生 TaskRunner 不取 descriptor；Trainer 直接 RPC Rollouter `get_replicas()`：`fully_async_trainer.py:167-173` |

正确扩展顺序应保留原生顺序，在 TaskRunner 初始化早期取得 GS handle，在 native replica 和 CE 创建完成后注册 topology；不能把 CE
创建提前到 Rollouter replicas 可用之前。

## 9. 空泡定义、模式表和图 5 校对

### 9.1 代码可观测量

| 量 | 代码事实 | 能否单独判断 replica 空泡 |
|---|---|---|
| LB `_inflight_requests[server_id]` | acquire +1、release -1：`llm_server.py:68-99` | 不能；admission 仍开放时只是瞬时零 |
| LB sticky map | request → server：`:64-90` | 不能；它不表示 future multi-turn 是否结束 |
| `pending_queue` | 尚未提交的 sample：`fully_async_rollouter.py:513-514,814-846` | 不能映射到具体 replica |
| `active_tasks` | sample coroutine task set：`:513-515,868-931` | 不能映射到具体 replica |
| MQ size | 已完成、待 Trainer 消费的 sample：`message_queue.py:55-119` | 不能映射到具体 replica |
| `paused` | 停止新的 sample admission：`fully_async_rollouter.py:848-893,1077-1099` | 必要上下文之一，不等于每个 replica 已空闲 |
| backend drain | `wait_for_requests_to_drain()`：`vllm_async_server.py:676-677,1056-1060` | 可确认 engine drain，但原生 LB 无两阶段摘流，存在新请求竞态 |

### 9.2 FullyAsync 模式公式

原文的公式可由代码直接验证：

```text
required_samples = ppo_mini_batch_size * require_batches
max_required_samples = int(required_samples * (staleness_threshold + 1)
                           * trigger_parameter_sync_step)
```

依据：`fully_async_rollouter.py:485-489,529-543`。暂停发生在 MQ 达 `max_queue_size` 或
`staleness_samples >= max_required_samples`：`:1077-1099`。每次参数同步后 staleness 重新基于 active task + MQ size 设置：
`:564-595`。

| 模式 | 原文结论 | 校对结果 |
|---|---|---|
| One-Step | 一个固定 generation batch | `[V]` `_async_gen_next_batch()` 构造一次 batch，Trainer await 完成后同步并启动下一 batch：`experimental/one_step_off_policy/ray_trainer.py:207-235,390-409` |
| `T=1,S=0` | 最大 `R` | `[V]` 代入公式成立 |
| `T>1,S=0` | 最大 `R*T` | `[V]` 代入公式成立 |
| `S>0,P=false` | 最大 `int(R*T*(1+S))` | `[V/M]` 上界和 pause 已实现；CE 同步仍会 abort（`checkpoint_engine/base.py:470-484`），client 因 P=false 不 retry，不能把 abort 完成等同于自然 drain |
| `S>0,P=true` | 同上界，参数同步可 abort/resume | `[V/M]` client 重试见 `fully_async_rollouter.py:98-149`；资源回收原因和同版本限制未实现 |

图 5 中 `Rollouter/Trainer → TaskRunner → GS`、`LB → GS`、`GS → TaskRunner` 全是 `[D]`。verl 原生只有 TaskRunner
持有 Trainer/Rollouter handles，没有反向主动上报 API：`fully_async_main.py:41-44,77-103`。

## 10. 图 6、图 7：捐赠和受赠创建时序校对

### 10.1 donor 时序

| 步骤 | 校对 | 代码锚点/缺口 |
|---|---|---|
| GS 发 DONATE | `[D]` | 模拟器只有 `InferScheduler.reclaim(num_instances)`：`interfaces.py:9-16` |
| LB ACTIVE→DRAINING | `[D]` | 原 LB remove 立即 pop：`llm_server.py:115-126` |
| 等 per-replica inflight=0 | `[M]` | count 可读：`:128-143`；缺少 drain 状态和原子 fence |
| CE 排除 replica | `[M]` | `remove_replicas()` 已有：`checkpoint_engine/base.py:422-429`；缺少同步并发保护 |
| Server level-2 sleep | `[D]` | STANDALONE skip：`vllm_async_server.py:625-635` |
| 返回 placement proof | `[M]` | launch 时能读 node/GPU：`:976-989`，但没有持久 descriptor/PG provenance API |
| 发布 lease | `[D]` | 模拟器 worker/task state 无 slot/lease epoch：`models.py:7-58` |

### 10.2 borrower 时序

| 步骤 | 校对 | 代码锚点/缺口 |
|---|---|---|
| GS ASSIGN 到 borrower TaskRunner | `[D]` | 生产协议无实现 |
| 运行期创建 STANDALONE Server | `[D]` | 原 `FullyAsyncLLMServerManager.add_replicas()` 只激活初始化时预注册的 hybrid replicas：`fully_async_rollouter.py:153-275` |
| 绕过 `init_standalone()` | `[D]` 且方向正确 | 原方法必然创建新 ResourcePool/PG：`replica.py:189-226` |
| borrower CE receiver | `[D]` | 可复用 `CheckpointEngineWorker.update_weights()`：`checkpoint_engine/base.py:278-325` |
| CE add PENDING_SYNC | `[M]` | 原 add list 已有：`:414-420`；PENDING_SYNC/version/epoch 均需新增 |
| 创建后不立即接流 | `[D]` 安全约束 | 原 LB add 后立即可 acquire：`llm_server.py:68-91,100-113`，必须扩展 hidden/SYNCING |
| 下一原生同步激活 | `[M]` | 原同步点和 replica 遍历可复用：`fully_async_trainer.py:501-524`、`checkpoint_engine/base.py:470-515`；ACK→LB 激活需新增 |

“V3.2 没有 V3 历史快照”是设计前提，不是 verl 类型；在不引入版本化外部存储时，等待下一原生同步是成立的保守策略。

## 11. 图 8：强制回收与 request continuation 校对

| 能力 | 现状 | 代码依据 |
|---|---|---|
| 全 replica abort | `[V]` | `RolloutReplica.abort_all_requests()`：`replica.py:273-279`；vLLM 聚合：`vllm_async_server.py:1062-1079` |
| abort 后暂停/恢复 engine | `[V]` | vLLM pause 逻辑：`:679-715`；resume：`:1081-1083` |
| 返回 aborted stop reason | `[V]` | `generate()`：`:547-580` |
| partial token 累积和 prompt prefix 重试 | `[V]` | `FullyAsyncLLMServerClient.generate()`：`fully_async_rollouter.py:91-149` |
| abort 必然携带 partial tokens | **不保证** | empty output 返回空 token：`vllm_async_server.py:547-556` |
| 仅 partial 配置开启时重试 | `[V]` | `fully_async_rollouter.py:141-145` |
| RESOURCE_RECLAIM reason/preemption ID | `[D]` | 原 TokenOutput 只有 stop_reason/extra_fields：`replica.py:39-51` |
| 同版本目的 replica 筛选 | `[D]` | 原 LB 不保存版本：`llm_server.py:60-91` |
| DRAINING/route token/EVACUATED | `[D]` | 原 LB 无状态机且 remove 立即删除：`:115-143` |
| 原 coroutine 内重试 | `[V/M]` | client while loop可复用：`fully_async_rollouter.py:98-145`；调度原因条件需新增 |
| 放回 MQ/pending/TransferQueue | `[V]` 原生没有这样做 | request 仍在 client coroutine；MQ 只接收完整 sample：`fully_async_rollouter.py:933-950` |

因此图 8 是 `[M]`：abort、partial accumulation、重新 acquire 的底层骨架已存在；其余控制面事务全部 `[D]`。主模拟器也不能作为
busy reclaim 的实现依据，因为 `execute_plan()` 只在 `idle_instances > 0` 时 reclaim：
`src/multi_rl_task_scheduler/group_scheduler.py:212-237`。

## 12. 图 9、图 10：参数同步时序校对

### 12.1 图 9：原生真实时序

`CheckpointEngineManager.update_weights()` 当前实际执行：

```text
abort all replicas
→ flatten current self.replicas[*].workers into temporary RayWorkerGroup
→ release KV cache（当前 vLLM 实现仍是 TODO/no-op）
→ prepare + build topology + init process group
→ trainer send_weights and rollout receive/apply
→ finalize
→ resume KV cache（当前 vLLM 实现仍是 no-op）
→ resume generation
```

依据：`checkpoint_engine/base.py:470-515`；vLLM KV 方法：`vllm_async_server.py:644-655`。

### 12.2 图 10：目标设计时序中各步骤

| 步骤 | 校对 | 代码依据/缺口 |
|---|---|---|
| 原生 Trainer sync trigger | `[V]` | FullyAsync：`fully_async_trainer.py:487-524`；One-Step：`separation/ray_trainer.py:645-650` |
| 遍历显式 replicas | `[V]` | `checkpoint_engine/base.py:485-489` |
| 同一 topology 包含 native+borrowed | `[M]` | 只要 borrowed 提供合法 worker handles，列表机制可表达；当前无 borrowed worker 实现 |
| add/remove replicas | `[V]` 基础列表操作 | `checkpoint_engine/base.py:414-429` |
| membership lock/snapshot epoch | `[D]` | 原 manager 无锁，直接修改 list |
| CE→LB fence | `[D]` | CE 不持有 LB handle |
| per-replica installed-version ACK | `[M]` | 原生 `ray.get` 等待全部 update RPC 成功，能作为 ACK 基础：`checkpoint_engine/base.py:322-325,498-502`；但没有按 replica 存储 version、LB commit 或回滚 |
| committed rollout version | `[D]` | FullyAsync Trainer 有 `current_param_version`，但 LB/CE 无 committed state：`fully_async_trainer.py:139-145` |
| all-or-nothing rollback | `[D]` | `asyncio/ray.get` 异常会冒泡，无多副本版本回滚 |
| NCCL rebuild | `[V]` 可配置能力 | `nccl_checkpoint_engine.py:115-125,146-157,200-221` |
| NIXL 动态链 topology | `[V]` 可重建 | `nixl_checkpoint_engine.py:288-369` |

原文“所有 READY replica 版本相同”是必须新增的安全不变量，不是当前 CE/LB 的保证。`global_steps` 会传给 ServerAdapter，vLLM Server
把它放进输出 `extra_fields`：`checkpoint_engine/base.py:323-325`、`vllm_async_server.py:558-602,672-674`；RPC 成功完成是隐式同步
barrier，但不是持久化的 per-replica version/路由事务。

## 13. 三视图状态表和核心不变量校对

三视图状态表整体是 `[D]` 的目标状态机。现有代码只提供以下局部事实：

- `[V]` Ray/PG owner：native ResourcePool/PG/worker actor；
- `[V]` LB active server 集合和 inflight count；
- `[V]` CE 当前 Python `replicas` list；
- `[S]` 模拟器 worker/task 分配、ACTIVE/SLEEPING 状态；
- `[D]` slot lease、DRAINING、PENDING_SYNC、membership/routing/lease epoch 和跨视图 CAS。

| 不变量 | 性质 | 代码锚点 |
|---|---|---|
| 一个 slot/epoch 只有一个 HBM 使用者 | `[D]` | 模拟器 `SleepRegistry` 的单 active instance/worker 可借鉴：`sleep_registry.py:118-194,267-291` |
| `LB_READY ⊆ CE_SYNC_READY` | `[D]` | 需扩展 LB `add_servers()` 和 CE update completion |
| donated native 不在 donor LB/CE | `[M]` | 原 LB/CE 均有 remove：`llm_server.py:115-126`、`checkpoint_engine/base.py:422-429`；原子性需新增 |
| borrowed sync ACK 前不路由 | `[D]` | 原 LB add 即可路由，必须新增 hidden/SYNCING |
| GS command 不等于 sync trigger | `[D]` 架构约束 | 保留 `_fit_update_weights()` 入口：`fully_async_trainer.py:501-524` |

## 14. 生命周期、故障和第一阶段范围校对

这些内容是目标设计约束，不是现有实现。每项的代码落点如下。

| 目标 | 性质 | 主要落点 |
|---|---|---|
| donor PG/session 存活约束 | `[D]` | TaskRunner session + GS lease；原生 `RolloutReplica.resource_pool/workers` 可作为 provenance |
| donor 仅 sleep、不销毁 | `[D]` | 实现 STANDALONE level-2 sleep/wake；模拟状态参考 `sleep_registry.py:196-239` |
| borrower 销毁不碰 donor PG | `[D]` | borrowed replica 专用 teardown，不能调用 donor ResourcePool 生命周期 |
| 多节点 ordered placement | `[M]` | 原生 worker 顺序和 node slicing：`vllm_async_server.py:976-998`；需持久化 rank mapping |
| task/session/lease epoch 故障失效 | `[D]` | GS/TaskRunner 协议新增；模拟器只有 state version：`group_scheduler.py:97-121` |
| 同 topology 第一阶段 | `[D]` 范围选择 | 可用 `RolloutReplica.world_size/nnodes/gpus_per_replica_node` 校验：`replica.py:103-117` |
| NCCL rebuild_group=true | `[V]` 后端参数存在 | `nccl_checkpoint_engine.py:115-125,146-157` |
| 不做 DDR 历史快照 | 方案范围 | verl 是否存在其他 backend 不影响本方案第一阶段约束 |

## 15. 已反写到 STANDALONE.md 的修订

1. 所有图统一在节点名前标 `[V]/[S]/[D]`，混合调用边标 `[M]`。
2. “原生运行图”和“目标设计图”明确分开；目标图不能用现在时暗示已经实现。
3. 原生部署图把 trainer-side CE 画进 Training Worker，而不是独立进程/Actor。
4. 原生 STANDALONE 图注明 PG bundle 预留 1 GPU、CheckpointEngineWorker actor 申请 0.5 GPU、Server Actor 不申请 Ray
   GPU但显式使用 worker GPU ID。
5. donor/borrower 图注明完整 `[D]`，并加 PoC gate。
6. 初始化图删除不存在的“native descriptor 返回 TaskRunner”和“CE 通知 LB READY”现状边，改成目标扩展虚线。
7. 空泡表保留公式，但把 “有限工作窗口”改成“admission 上界”，避免把已接纳 active task 误认为已经完成。
8. sleep 条件明确分成 `drained` 与 `HBM_RELEASED`；原生只提供前者且 STANDALONE sleep no-op。
9. 强制回收明确为 `[M]`：partial retry 骨架可复用，资源抢占原因、同版本目的 replica 和两阶段 LB 均需新增。
10. 参数同步图先画原生真实流程，再画目标 fence/ACK；注明 KV release/resume 在当前 vLLM adapter 中也是 no-op。
11. `effective_replicas` 定义为目标封装；原生名称只是 `CheckpointEngineManager.replicas`。
12. 第一阶段增加四个前置验证门：STANDALONE level-2 sleep、borrowed CE worker 无重复 Ray GPU reservation、同 GPU 双进程
    CUDA/vLLM 隔离、动态 NCCL topology 和失败恢复。
13. 把 MultiTask Trainer/Rollouter 标为逻辑角色，补充 Ray ActorClass 不能直接普通继承，以及 manager factory/RPC 注入方案。
14. 补充 TaskRunner 运行期命令的并发缺口：FullyAsync 长期阻塞、One-Step Trainer 局部变量及事务锁要求。

上述修订已反写到 `STANDALONE.md`。更新后的主文档可以作为设计评审稿；它仍不是实现说明书。每个 `[D]` 项在进入编码前需要形成
接口 schema、状态机和最小 PoC 测试。
