# verl v0.8.0 可复用接口与扩展边界

> 基线：v0.8.0 tag `7aed6b23`。本文只保留与 GS、TaskRunner、`MultiTaskPPOTrainer`、
> `MultiTaskLLMServerManager`、扩展 LB 和版本化 DDR 权重存储直接相关的代码结论。

## 1. 当前对象位置

新 sync 路径的控制关系是：

```text
run_ppo()
└─ TaskRunner Ray Actor                  # 每任务 single controller
   └─ PPOTrainer                         # controller 内普通 Python 对象
      ├─ actor_rollout_wg                # 训练 worker group
      ├─ LLMServerManager                # 每任务普通 Python 对象
      │  ├─ RolloutReplica[]
      │  └─ GlobalRequestLoadBalancer    # Ray Actor
      ├─ CheckpointEngineManager         # 权重同步与 replica 集合
      └─ AgentLoopManager                # 生成请求入口
```

关键证据：

- `main_ppo.py:52-102`：`run_ppo()` 创建一个 `TaskRunner` Ray Actor，并阻塞等待 `runner.run()`。
- `main_ppo_sync.py:501`：新 sync `PPOTrainer` 是 controller 内普通对象，不是 Ray Actor。
- `main_ppo_sync.py:712-714`：`init_workers()` 直接调用内置 `LLMServerManager.create(...)`。
- `main_ppo_sync.py:732-736`：随后用 manager 的 replica 列表构造 `CheckpointEngineManager`。

因此，GS actor 可以被所有任务共享，但 LLMServerManager 本身不能作为 GS 的远程回调目标。目标方案
复用 `TaskRunner` 的 Ray Actor 边界：TaskRunner 持有 GS handle，GS 保存每个已注册 TaskRunner handle，
命令沿 GS→TaskRunner→trainer→manager 执行。`GlobalRequestLoadBalancer` 本身是 Ray Actor，可持有 GS
handle 并直接上报 rollout 空闲事实。

## 2. 可以直接复用的能力

### 2.1 Ray 单例

verl 已有 named detached actor 的用法：`TrajectoryTracker.options(name=..., get_if_exists=True, lifetime="detached")`。子仓的 GS 可以采用同类模式，并显式固定 Ray namespace：

```python
GroupScheduler.options(
    name="verl_group_scheduler",
    namespace="verl-multi-task",
    get_if_exists=True,
    lifetime="detached",
).remote(config)
```

所有任务必须连接同一 Ray cluster 且使用约定 namespace。`task_id` 和 `session_id` 不能依赖 Ray job id，因为任务重启后 job id 会变化。

### 2.2 单任务请求路由

`GlobalRequestLoadBalancer` 已支持：

- `add_servers({address: handle})`；
- `remove_servers(server_ids)`；
- sticky session 失效后的重新选择；
- in-flight 计数和状态查询。

它适合继续作为每任务内部的流量路由器。目标扩展需要在其 acquire/release 账本中增加 generation
级输入取尽判定，并注入 GS handle 与 `TaskContext`，直接上报 `STEP_IDLE_REPLICAS`。GS 不应直接修改
LB；所有摘流、创建和回收命令仍经 TaskRunner 进入 `MultiTaskLLMServerManager`。

### 2.3 replica 控制原语

`RolloutReplica` 已提供：

- `sleep()` / `wake_up()`；
- `abort_all_requests()` / `resume_generation()`；
- `release_kv_cache()` / `resume_kv_cache()`；
- server handle/address 和 worker handles。

这些能力足以实现“预创建 replica 的 activate/deactivate”原型，但不等同于物理创建/销毁。

### 2.4 权重同步集合更新

`CheckpointEngineManager` 已有 `add_replicas()` 和 `remove_replicas()`，并在 `update_weights()` 时读取当前列表。它只维护列表，不负责同步更新 load balancer 或销毁实例，因此必须由扩展 manager 统一编排。

### 2.5 Mooncake checkpoint backend

v0.8.0 已注册 `MooncakeCheckpointEngine`，可复用 `TransferEngine`、memory registration 和
`transfer_sync_read/write`。但其当前语义仍是一次权重同步：trainer 与 rollout 同时构建临时
`StatelessProcessGroup`，并发 send/receive；buffer 默认在 CUDA device，缺少版本化 DDR manifest、
独立 reader、lease 和 GC。非 naive `update_weights()` 还会 abort 全部 replicas、释放全部 KV，并为
全部 rollout workers 重建通信组，不适合只给 rollout 中新增的一个 replica 加载。因此它是目标 weight
store 的传输基础，不是现成的 DDR snapshot store。

### 2.6 可参考的 experimental 实现

`FullyAsyncLLMServerManager` 展示了：

- 预注册 HYBRID replicas；
- 运行时把 server 加入/移出 load balancer；
- 维护 alive replica/address 映射；
- trainer 侧在切换阶段进行 update weights、LB activate/deactivate 和 sleep。

它是生命周期编排参考，但当前 `add_replicas/remove_replicas` 只改变活跃集合和 LB，不创建或销毁 server；同时该代码绑定 fully-async/validation 场景，不能直接作为 sync 方案。

## 3. v0.8.0 的直接缺口

### 3.1 trainer class 不可注入

`run_ppo(..., task_runner_class=...)` 可以注入自定义 remote ActorClass，但 sync `TaskRunner.run()`
直接构造 `PPOTrainer`。同时，模块导出的 `TaskRunner` 已经被 `@ray.remote` 装饰，不能作为普通
Python 基类直接继承。

子仓原型需要提供新的 remote `MultiTaskTaskRunner` 并移植较短的 controller 编排。更适合 RFC 的
方式是让 TaskRunner 通过 trainer FQN/factory 选择 `MultiTaskPPOTrainer`，或额外暴露未 remote 的
可继承 implementation。

### 3.2 LLMServerManager 不可配置替换

新 sync trainer 直接引用并实例化内置 `LLMServerManager`，没有 `llm_server_manager_class` 配置项。`agent_loop_manager_class` 虽可用 FQN 替换，但不能替换实例生命周期管理器。

子仓原型可以覆写 `PPOTrainer.init_workers()`，但会复制大段上游初始化逻辑。更适合 RFC 的改法是新增一个 manager factory/FQN，并把 `config/worker_group/resource_pool` 原样传给外部类。

初始化阶段不需要改变这项行为：`MultiTaskLLMServerManager` 继续复用父类逻辑，按本任务配置创建
初始 replicas。缺口只影响运行期 GS 实时扩缩。

### 3.3 base manager 只支持一次性初始化

`LLMServerManager._initialize_llm_servers()`：

1. 从固定 `worker_group.world_size` 计算 replica 数；
2. 一次性构造全部 `RolloutReplica`；
3. 一次性启动 server；
4. 一次性创建 LB。

base 类没有 add/remove/shutdown API，内部 `rollout_replicas/server_handles/server_addresses` 也没有统一事务保护。

### 3.4 HYBRID placement 固定

`RolloutReplica.init_hybrid(worker_group)` 按：

```python
workers[world_size * replica_rank : world_size * (replica_rank + 1)]
```

选择 worker。这只适用于“从本任务固定 worker group 顺序切片”。GS 给出的任意 worker placement、跨任务 worker lease 或重用历史 binding 都无法表达。

需要一个接受显式 `worker_handles` 的初始化入口，并校验：数量、节点拓扑、GPU ID、lease owner、任务 session 和 replica 唯一名。

### 3.5 没有通用 teardown

`RolloutReplica` 没有 `shutdown/close/destroy`；vLLM async server 也没有被 manager 统一调用的 teardown。LB 的 `remove_servers()` 仅停止新请求路由，不会：

- 等待或中断在途请求；
- sleep/关闭 vLLM engine；
- 终止 Ray server actor 和 mp worker 子进程；
- 清理 socket、server name 和本地列表；
- 释放或归还 worker lease。

因此训练中“真正销毁”不能靠从 Python list 删除，也不应直接用未经封装的 `ray.kill` 代替。

### 3.6 缺少训练侧脱离读路径的版本化权重存储

HYBRID naive backend 依赖训练 worker 内的 `ServerAdapter` 与既有 vLLM worker 的 IPC 映射。当前
Mooncake backend 虽可远程读 buffer，但 `CheckpointEngineManager.update_weights()` 仍要求 trainer 和
rollout 同时加入同步拓扑。动态 replica 在 rollout 中创建时，训练侧往往不可用，不能以“重新唤醒
trainer 并建一次 process group”作为前提。

目标是 trainer 在 policy commit 时把 rollout-format 权重发布为版本化 immutable DDR snapshot，动态
replica 按 `WeightSnapshotRef` 直接 DDR→HBM。需要回答：

- 谁负责创建对应 ServerAdapter/IPC endpoint；
- 同一 worker 是否允许多个 task/replica adapter；
- 谁把 sharded training weights 转成 rollout 可消费的 manifest/load plan；
- snapshot 如何原子 commit、校验、pin 和 GC；
- 多个动态 replica 如何并发读取同一 version；
- policy version 推进时如何保留仍被 generation/command 引用的旧 snapshot；
- donor 训练和 borrower rollout 如何互斥；
- replica 归还或重建后如何清理旧权重和 IPC 资源。

这部分是 HYBRID PoC 的首要验证项，模拟器没有覆盖。详细协议见
[`08-versioned-ddr-weight-store.md`](./08-versioned-ddr-weight-store.md)。

### 3.7 多任务命名冲突

vLLM server actor 名主要由 backend、replica rank 和 node rank组成。多个任务共享 Ray namespace 时，需要使用已有 `name_suffix` 能力加入 `task_id/session_id`，否则同 rank server 可能冲突。ZMQ 路径已经包含 Ray job id，但跨任务 lease 和任务重启仍需显式 session fencing。

## 4. 建议的最小上游接口

### 4.1 manager 注入

```yaml
actor_rollout_ref:
  rollout:
    llm_server_manager_class: my_project.MultiTaskLLMServerManager
```

默认值保持内置类，关闭子仓能力时行为不变。

### 4.2 load balancer 注入与请求标识

为避免复制整个 manager 初始化过程，建议为 `GlobalRequestLoadBalancer` 增加 factory/FQN，并允许透传
可选的 GS handle、`task_id/session_id`。生成请求还需要稳定的 `GenerationKey(partition,
global_steps, dispatch_id)` 与 `(prompt_uid, rollout_index)`，LB 才能区分“首次取尽本 step 输入”和
普通的瞬时零负载。默认 LB 和未携带这些字段的路径保持原行为。

### 4.3 trainer lifecycle hook

`MultiTaskPPOTrainer` 需要把 `ROLLOUT_READY/ROLLOUT_RUNNING/ROLLOUT_DRAINED/TRAIN/WEIGHT_SYNC/EXITING`
通知 manager。资源命令在 rollout 中实时执行，因此 hook 不负责 poll 或 apply command，只负责提供
phase/policy-version 安全边界。TaskRunner 还需要两个线程安全的 trainer 窄接口：读取最近 committed
snapshot，以及执行一个 versioned schedule command。

### 4.4 replica 生命周期

建议把机制接口放入 verl，把策略留在子仓：

```python
class LLMServerManager:
    async def add_replicas(self, specs: list[ReplicaSpec]) -> list[ReplicaResult]: ...
    async def remove_replicas(self, replica_ids: list[str], *, drain: bool) -> list[ReplicaResult]: ...

class RolloutReplica:
    async def init_hybrid_workers(
        self, workers, *, name_suffix: str, weight_source=None
    ): ...
    async def shutdown(self, *, graceful: bool): ...
```

`ReplicaSpec` 应携带稳定 worker 标识、实际 actor handles 或可解析引用、拓扑、task/session、replica id 和期望 policy version。

### 4.5 版本化权重发布与加载

建议增加通用 hook，而不是把 Mooncake 写死进 trainer：

```python
class WeightSnapshotPublisher:
    def publish(self, policy_version: int) -> WeightSnapshotRef: ...

class WeightSnapshotLoader:
    async def load_into_replica(
        self, replica, snapshot_ref, expected_policy_version
    ) -> LoadResult: ...
```

trainer 在 policy commit 后、进入下一轮 rollout 前 publish；rollout worker 按 manifest 从 DDR 写入 HBM
并返回 loaded version/digest。publisher 成功后无需参与 reader 路径。verl 需要暴露 rollout-format
tensor stream 和 adapter load/verify hook；snapshot catalog、Mooncake backend 和 GC 留在子仓。
动态 replica 还应支持 deferred-weight 初始化，避免先从磁盘加载模型再被 DDR snapshot 覆盖。

### 4.6 原子更新顺序

扩容推荐顺序：

```text
校验 lease/placement
→ 创建 replica/server
→ 加入 CheckpointEngineManager
→ pin committed snapshot
→ 从 DDR 加载目标 policy version 到 HBM，并校验 digest
→ warmup
→ 最后加入 LB
→ 上报 committed
```

缩容推荐顺序：

```text
先从 LB 摘流
→ drain 或 abort 在途请求
→ 从 CheckpointEngineManager 移除
→ sleep/shutdown replica
→ 清理 actor/子进程/名称
→ 释放 worker lease
→ 上报 committed
```

中途失败必须返回结构化结果，并能从实际对象重新构建状态；不能只回滚内存列表。

## 5. 侵入等级

| 等级 | 能力 | 是否可只在子仓完成 |
|---|---|---|
| L0 | GS 单例、协议、TaskRunner 注册/注销与主动 heartbeat、只读状态与调度计算 | 是 |
| L1 | 自定义 runner/trainer/manager/LB、LB→GS 空闲上报、GS→TaskRunner 命令链路 | 是，但需移植部分 controller/init 编排 |
| L2 | 通用 manager/LB factory、TaskRunner endpoint、phase/request-context/policy-publish hook | 需要小型 verl 扩展，适合 RFC |
| L3 | versioned DDR publish/load、rollout 中按任意 placement 实时 create/destroy HYBRID replica | 需要 verl 权重导出/加载和生命周期改造，侵入最高 |

第一阶段应按 L0→L1→L2→L3 逐层验证，避免在协议尚未稳定时先改动 vLLM 生命周期核心。
