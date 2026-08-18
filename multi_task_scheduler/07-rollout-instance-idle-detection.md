# Rollout 推理实例空闲识别方案

> 状态：当前方案。基线为 verl v0.8.0，commit `7aed6b23`。
>
> 本文只回答两个问题：当前 step 的所有 rollout trajectories 何时都已进入推理，以及如何据此识别并
> 稳定摘流空闲推理实例。TaskRunner 通用控制面、GS 调度算法、权重同步和动态创建/销毁细节不在本文
> 展开；版本化 DDR 权重路径见
> [`08-versioned-ddr-weight-store.md`](./08-versioned-ddr-weight-store.md)。

## 1. 结论

verl 当前没有“推理实例从公共 batch 队列取 sample”的动作。实际行为是：所有 prompt 先被分发到
`AgentLoopWorkerTQ`，每个 trajectory 调用 `LLMServerClient.generate()` 时，再由任务内
`GlobalRequestLoadBalancer` 动态选择一个 rollout server。

因此，“当前 step 所有 sample 都被推理实例取走”需要定义为：

```text
BATCH_INPUT_EXHAUSTED
= 当前 generation 内，每个预期 trajectory_id 都至少完成过一次 LB acquire
```

这里的“取走”精确定义为：LB 已为 trajectory 建立 routing lease、选定 server，并把该 server 的
in-flight 加一。它不表示 vLLM 已完成推理；`server.generate.remote()` 紧随 acquire 发起。即使两者之间
存在极短调度间隙，该 trajectory 已计入目标 server 的 in-flight，不会把目标误判为空闲。

其中：

```text
trajectory_id = (prompt_uid, rollout_index)
rollout_index  = verl AgentLoopWorkerTQ 中名为 session_id 的 range(n) 下标
```

当前代码没有这个聚合信号，需要扩展任务内 LB。扩展后：

1. trainer 在分发 batch 前向 LB 注册本 step 的 `expected_trajectory_ids`；
2. 每个 trajectory 首次 acquire 时，LB 去重记录 `trajectory_id`；
3. `observed == expected` 时，LB 将当前 step 标记为 samples exhausted；
4. 对 `SINGLE_GENERATE` batch，LB 把满足 `input_exhausted && inflight == 0` 的实例直接上报 GS；
5. GS 决定回收后，manager 在 LB 内原子执行“确认仍为零 + 从路由集合摘除”，再 drain/sleep/destroy。

对 single-turn Agent，`BATCH_INPUT_EXHAUSTED` 后不会再有正常的新 acquire，实例会随剩余请求完成而
逐步空闲，这正是“该 step 样本已完全消耗干净”的精确定义。对 multi-turn Agent，它只代表所有
trajectory 已完成首次 acquire；工具调用后仍可能再次 generate，因此不能按同一语义向 GS 报告
`STEP_IDLE_REPLICAS`。multi-turn 要么等 `BATCH_TERMINATED`，要么另行使用候选空闲+原子摘流协议。

## 2. 代码基线

### 2.1 当前生成链路

```text
PPOTrainer.step()
├─ 为 prompt 分配 uid，写入 global_steps
├─ AgentLoopManagerTQ.generate_sequences(batch)
│  ├─ ReplayBuffer: uid → running
│  ├─ batch 按 AgentLoopWorker 数量切块
│  └─ AgentLoopWorkerTQ.generate_sequences.remote(chunk)
│     └─ 每个 prompt 创建后台 _run_prompt task
│        └─ 按 n 创建 (uid, rollout_index) trajectory tasks
│           └─ AgentLoop.run()
│              └─ LLMServerClient.generate()
│                 ├─ LB.acquire_server()
│                 ├─ await vLLMHttpServer.generate.remote()
│                 └─ LB.release_server()
└─ ReplayBuffer.sample(global_steps)
   └─ 等到本 step 不再有 running prompt
```

`AgentLoopManagerTQ.generate_sequences()` 中的 `ray.get()` 只等待 worker 把 `_run_prompt` 后台任务创建
出来，不等待 trajectory 执行到 LB acquire。`ReplayBuffer.sample()` 返回又太晚：此时所有 prompt 已经
finished 或 failure，rollout 尾部可渐进回收的窗口已经结束。

### 2.2 sample 的三个不同基数

```text
prompt 数       = 注册到 ReplayBuffer 的 uid 数
trajectory 数   = Σ 每个 prompt 的实际 n
output row 数   = trajectory 最终产生的 AgentLoopOutput 数
```

实际 `n` 的来源：

- normal train：缺省为 `rollout.n`；
- validation：缺省为 `val_kwargs.n`；
- ReMax sampled prompt：显式 `__rollout_n__=rollout.n`；
- ReMax baseline prompt：显式 `__rollout_n__=1`；
- dataset/recipe 还可以逐 prompt 动态覆盖 `__rollout_n__`。

所以不能使用 `len(batch) × rollout.n` 推导总 trajectory 数，必须逐 prompt 读取实际 `n`。

### 2.3 当前已有和缺失的信号

| 信号 | 当前是否存在 | 精确含义 |
|---|---:|---|
| `BATCH_PROMPTS_DISPATCHED` | 是 | AgentLoopManager 的 `ray.get()` 返回；所有 prompt 后台任务已创建 |
| `BATCH_INPUT_EXHAUSTED` | 否 | LB 内部状态：所有预期 trajectory 至少完成一次 LB acquire |
| `STEP_IDLE_REPLICAS` | 否 | single-generate step 已 input-exhausted，且所列 server 当前 in-flight 为 0 |
| `BATCH_TERMINATED` | 是 | ReplayBuffer 不再有本 step 的 running prompt，可能 finished 或 failure |

## 3. `BATCH_INPUT_EXHAUSTED` 判断机制

### 3.1 唯一 generation 标识

不能只用 `global_steps` 标识 batch。`_validate()` 会在同一个 `global_steps` 下迭代多个 validation
batches。每次 `generate_sequences()` 使用独立标识：

```text
GenerationKey(
    partition_id,       # train / val
    global_steps,
    dispatch_id,        # 每次分发唯一的单调序号或 UUID
)
```

### 3.2 计算预期 trajectory 集合

trainer 在实际 rollout batch 构造完成后计算：

```python
expected_trajectory_ids = {
    (prompt_uid, rollout_index)
    for prompt_uid, rollout_n in zip(batch["uid"], per_prompt_rollout_n, strict=True)
    for rollout_index in range(rollout_n)
}
```

在调用 `AgentLoopManagerTQ.generate_sequences()` 前必须先完成：

```text
await LB.begin_generation(
    generation_key,
    expected_trajectory_ids,
    agent_capability,
)
```

注册必须 await，保证任何 acquire 都不会早于 generation state 创建。

### 3.3 将 trajectory context 传到 LB

`AgentLoopWorkerTQ._run_prompt()` 已经向 `_run_agent_loop()` 传入：

```text
uid
session_id     # 这里实际是 rollout_index
global_steps
```

MultiTask trainer 还需给 batch 注入内部 `__generation_key__`。MultiTask AgentLoopWorker 在进入具体
AgentLoop 前，把这些字段写入 task-local `ContextVar`，并从传给 concrete AgentLoop 的 kwargs 中移除
内部字段：

```python
context = RolloutRequestContext(
    generation_key=generation_key,
    trajectory_id=(prompt_uid, rollout_index),
)

token = rollout_request_context.set(context)
try:
    return await agent_loop.run(...)
finally:
    rollout_request_context.reset(token)
```

同一 AgentLoopWorker 中每个 trajectory 都是独立 asyncio task，`ContextVar` 不会在并发 trajectories
之间互相覆盖。`MultiTaskLLMServerClient.generate()` 读取该 context，并传给 LB acquire。

不能只用 sticky `request_id` 代替 `trajectory_id`：自定义 AgentLoop 可能在同一个 trajectory 内创建
多个 request id，或在重试时更换 request id。

### 3.4 LB 计数

LB 为每个 active generation 保存：

```text
expected_trajectory_ids
observed_trajectory_ids
input_exhausted
agent_capability
server_id → inflight
server_id → server_load_version
routing_lease_id → lease state
```

首次 acquire 的核心逻辑：

```python
def acquire_server(request_id, request_context):
    generation = generations[request_context.generation_key]
    trajectory_id = request_context.trajectory_id

    if trajectory_id not in generation.expected_trajectory_ids:
        report_protocol_error("UNKNOWN_TRAJECTORY", trajectory_id)
    elif trajectory_id not in generation.observed_trajectory_ids:
        generation.observed_trajectory_ids.add(trajectory_id)

    server = select_sticky_or_least_loaded(request_id)
    server.inflight += 1
    server.server_load_version += 1
    lease = create_routing_lease(server, request_context)

    if (
        not generation.input_exhausted
        and generation.observed_trajectory_ids == generation.expected_trajectory_ids
    ):
        generation.input_exhausted = True
        if generation.agent_capability == SINGLE_GENERATE:
            report_current_step_idle_replicas_to_gs(generation)

    return server.handle, lease
```

后续 multi-turn acquire 使用同一个 `trajectory_id`，不会重复增加 observed 数量。未知 trajectory、重复
release 和不存在的 generation 都记录协议错误，但不能静默污染计数。

## 4. 完整例子：8 个 trajectories、3 个推理实例

假设训练 step 42：

```text
partition_id = train
global_steps = 42
dispatch_id  = train-42-0

prompt uids  = [p0, p1, p2, p3]
每个 prompt 的 rollout_n = 2
replicas = [R0, R1, R2]
agent capability = SINGLE_GENERATE
```

trainer 注册的预期集合为：

```text
expected = {
  (p0, 0), (p0, 1),
  (p1, 0), (p1, 1),
  (p2, 0), (p2, 1),
  (p3, 0), (p3, 1),
}
expected_count = 8
```

所有 trajectory 的首次 acquire 过程如下。为突出判断点，表中假设 acquire 期间请求尚未完成：

| 时刻 | 首次 acquire 的 trajectory | LB 选择 | observed/expected | R0/R1/R2 in-flight | batch 状态 |
|---|---|---|---:|---|---|
| T1 | `(p0,0)` | R0 | 1/8 | 1/0/0 | accepting |
| T2 | `(p0,1)` | R1 | 2/8 | 1/1/0 | accepting |
| T3 | `(p1,0)` | R2 | 3/8 | 1/1/1 | accepting |
| T4 | `(p1,1)` | R0 | 4/8 | 2/1/1 | accepting |
| T5 | `(p2,0)` | R1 | 5/8 | 2/2/1 | accepting |
| T6 | `(p2,1)` | R2 | 6/8 | 2/2/2 | accepting |
| T7 | `(p3,0)` | R0 | 7/8 | 3/2/2 | accepting |
| T8 | `(p3,1)` | R1 | 8/8 | 3/3/2 | `BATCH_INPUT_EXHAUSTED` |

T8 在 LB Actor 内完成以下原子状态更新：

```text
observed 增加 (p3,1)
→ observed == expected
→ input_exhausted = true
→ 生成当前 server 快照 {R0:3, R1:3, R2:2}
→ 当前没有 inflight=0 的实例，不上报 STEP_IDLE_REPLICAS
```

这就是识别“当前 step 的所有 sample 都已经被推理实例取走”的准确时刻。它不是
`AgentLoopManager.generate_sequences()` 返回的时刻，也不是所有请求完成的时刻。

之后请求逐步完成：

| 时刻 | release | R0/R1/R2 in-flight | 推理侧事件 |
|---|---|---|---|
| T9 | R2 的两个请求完成 | 3/3/0 | `LB → GS: STEP_IDLE_REPLICAS([R2])` |
| T10 | R0 的三个请求完成 | 0/3/0 | `LB → GS: STEP_IDLE_REPLICAS([R0])` |
| T11 | R1 的三个请求完成 | 0/0/0 | `LB → GS: STEP_IDLE_REPLICAS([R1])` |

因此 single-turn 场景中，T8 之后空闲实例会随 batch 尾部完成逐步增加。GS 可以在 T9 直接收到 LB 的 R2 空闲
事件后立即尝试回收 R2，不必等到 T11 或 `ReplayBuffer.sample()` 返回。

若这个例子使用 tool Agent，T8 只表示 8 个 trajectories 都完成过首次 acquire。某个 trajectory 可能
随后执行 tool，并再次调用 generate；因此不能把 T8 理解为“未来请求全部发出”。

## 5. 推理实例空闲状态

### 5.1 两层空闲语义

```text
STEP_IDLE_OBSERVED
= generation 为 SINGLE_GENERATE
  && generation.input_exhausted
  && server.inflight == 0

RECLAIMABLE
= server 已在 LB 内原子标记 DRAINING 并从 acquire 候选集合排除
```

`STEP_IDLE_OBSERVED` 表示目标实例已经消耗完本 step 路由给它的 samples，但仍不是资源账本提交条件。
事件可能延迟，或在 GS 下发回收前发生显式重试；执行侧仍需原子复核。

### 5.2 single-turn

当同时满足：

```text
agent_capability == SINGLE_GENERATE
generation.input_exhausted == true
server.inflight == 0
```

可以推导该 server 已自然处理完当前 batch 路由给它的请求，不会再接到本 batch 的正常新请求。LB
此时直接向 GS 上报。执行回收前仍调用 `try_mark_draining()`，处理事件延迟、重试和并发命令。

### 5.3 multi-turn

即使 `input_exhausted == true && inflight == 0`，unfinished trajectory 仍可能在工具处理后再次 acquire，
所以不满足“该 step samples 已完全消耗干净”的上报语义。当前协议不向 GS 报告
`STEP_IDLE_REPLICAS`；只有 `BATCH_TERMINATED` 后才确定没有 future turn。若未来需要更激进的
multi-turn 候选回收，应另设事件并保持原子摘流，不能复用本事件名称。

同一个 batch 若混合多个 `agent_name`，只有全部 trajectories 都声明 single-generate，batch capability
才是 `SINGLE_GENERATE`；存在 tool/multi-turn 或未知 AgentLoop 时统一使用 `MAY_GENERATE_AGAIN`。

## 6. 原子摘流

收到空闲事件后不能直接 sleep/destroy。正确顺序：

```text
1. GS 根据 LB 直报的 STEP_IDLE_REPLICAS 生成 RECLAIM
2. GS → TaskRunner → trainer → MultiTaskLLMServerManager
3. manager → LB.try_mark_draining(server_id, expected_server_load_version)
4. LB 原子检查：
     state == ACTIVE
     inflight == 0
     server_load_version == expected
5. 检查成功：ACTIVE → DRAINING，并立即排除出路由候选
6. 检查失败：返回 BUSY/STALE，不执行资源释放
7. manager 调用 server.wait_for_requests_to_drain() 二次确认 engine
8. 更新 CheckpointEngineManager
9. sleep/destroy replica，释放 worker lease
10. 返回 CommandResult，GS 提交资源账本
```

检查零值和排除路由必须在同一个 LB Actor method 内完成。当前 verl `remove_servers()` 只是直接删除
字典项，不检查 in-flight，也没有 draining 状态，不能直接承担该协议。

建议新增：

```python
try_mark_draining(server_id, expected_server_load_version) -> DrainReservation
commit_remove(reservation_token) -> None
cancel_draining(reservation_token) -> None
```

`wait_for_requests_to_drain()` 只能确认 engine 当前请求排空，不能阻止新请求从 LB 进入，所以必须位于
LB 原子摘流之后。

## 7. 事件与通信路径

不增加新的 Actor：

```text
MultiTaskGlobalRequestLoadBalancer Actor
    │ holds GroupScheduler ActorHandle
    │ report_step_idle_replicas.remote(event)
    ▼
GroupScheduler Actor
    │ decide RECLAIM
    ▼
MultiTaskTaskRunner.apply_schedule_command.remote(command)
```

LB 持有 GS ActorHandle 和不可变 TaskContext，不持有 TaskRunner handle。它只能报告请求面事实，不能
直接修改 GS committed ledger，也不能触发 manager 生命周期方法。GS 收到事件后仍必须经 TaskRunner
执行规模调整。

### 7.1 上报条件

```text
generation.agent_capability == SINGLE_GENERATE
&& generation.input_exhausted == true
&& server.state == ACTIVE
&& server.inflight == 0
```

LB 在两个时刻检查并上报：

1. `input_exhausted` 从 false 变为 true 时，扫描并一次性上报当时已为零的所有 active servers；
2. 之后某 server release 使 in-flight 从 `1 → 0` 时，上报该 server。

这样不会漏掉在 input-exhausted 之前已经为零、之后不再发生状态转换的实例。同一
`generation_key/server_id/server_load_version` 只报告一次。

### 7.2 `STEP_IDLE_REPLICAS`

```text
task_id / task_session_id
generation_key
replicas[]:
  replica_id / server_id
  inflight_requests = 0
  server_load_version
agent_capability = SINGLE_GENERATE
input_exhausted = true
observed_at
```

`server_load_version` 按目标 server 的 acquire、有效 release、draining、remove/cancel 单调递增。其他
server 的负载变化不应使目标事件失效。它只用于请求面竞争校验，不替代 manager 生命周期事务提交后的
`state_version`。GS 按 task/session/generation/server/version 幂等去重，旧 session 的 LB 事件直接拒绝。

## 8. 需要扩展的组件

| 组件 | 调整 |
|---|---|
| `MultiTaskPPOTrainer` | 持有 TaskRunner 传入的 GS handle；构造唯一 GenerationKey；按实际 per-prompt n 计算 expected trajectories；在 dispatch 前 await LB 注册；terminal 时记录 missing/failure |
| MultiTask AgentLoopWorker | 将 generation key 和 `(prompt_uid, rollout_index)` 放入 task-local request context |
| `MultiTaskLLMServerClient` | acquire/release 使用 request context 和 routing lease |
| `MultiTaskGlobalRequestLoadBalancer` | 持有 GS handle/TaskContext；generation tracker、first-acquire 去重、in-flight/load version、step-idle 直报、原子 draining |
| `MultiTaskTaskRunner` | 注册/心跳/命令 endpoint；不转发 rollout 事件，不持有 manager |
| `MultiTaskLLMServerManager` | 执行原子摘流后的 drain、CE 更新和 sleep/destroy 事务 |
| `GroupScheduler` | 接收 LB 直报并校验 task/session；通过 TaskRunner 下发命令；根据 CommandResult 才提交资源账本 |

这些扩展都可位于独立子仓。由于 verl 的 `GlobalRequestLoadBalancer` 已经被 `@ray.remote` 包装，短期
PoC 可在子仓实现 API 兼容的替代 Actor；RFC 更适合让 verl 提供 LB implementation/factory 注入点。

## 9. 异常和 fencing

- trajectory 在首次 acquire 前失败：不进入 `BATCH_INPUT_EXHAUSTED`，因此不直报 step-idle；`BATCH_TERMINATED` 记录
  missing trajectory ids；
- acquire 重试或 multi-turn：按 trajectory id 去重，不重复增加 observed 数；
- release 是 fire-and-forget：routing lease 绑定 generation key，迟到 release 不读取可变的
  `current_generation`；
- 同一 global step 的多个 validation batches：dispatch id 不同，事件不能跨 batch 合并；
- 重复/迟到 release：按 routing lease id 去重；
- TaskRunner 重启：新 TaskContext 使用新 task session；GS 直接拒绝旧 LB session 的事件；
- generation terminal 后仍有未关闭 lease：标记异常并等待/核对，不能直接释放对应实例。

## 10. 最小验证用例

1. normal、ReMax、validation 的 expected trajectory 集合按实际 per-prompt n 正确生成；
2. 最后一个不同 trajectory 首次 acquire 时只把 generation 的 `input_exhausted` 置位一次；
3. input-exhausted 转换会直接向 GS 报告当时已经为零的 active servers，不漏报；
4. single-turn 在输入取尽后不再产生正常新 acquire，LB 按 release 顺序直接向 GS 报告逐步变零的实例；
5. multi-turn 的后续 acquire 不重复计数，且不会错误产生 `STEP_IDLE_REPLICAS`；
6. `STEP_IDLE_OBSERVED` 后发生竞争 acquire 时，`try_mark_draining` 返回 BUSY/STALE；
7. `try_mark_draining` 成功后，sticky 和非 sticky 新请求都不能进入目标 server；
8. trajectory 首次 acquire 前失败时，不误报输入取尽；
9. 同 global step 的多个 validation batches 和迟到 release 不串 generation；
10. 重复事件、重复 release、旧 task session 的 LB 直报均被 GS 幂等丢弃。

## 11. 代码索引

| 事实 | verl v0.8.0 位置 |
|---|---|
| prompt running/terminal barrier | `verl/trainer/main_ppo_sync.py:261-294` |
| AgentLoopWorker 创建 prompt/trajectory tasks | `verl/trainer/main_ppo_sync.py:304-377` |
| batch chunk 和 worker dispatch | `verl/trainer/main_ppo_sync.py:475-495` |
| ReMax 实际 batch 和 per-prompt n | `verl/trainer/main_ppo_sync.py:1688-1706` |
| trainer dispatch、sample、sleep | `verl/trainer/main_ppo_sync.py:1705-1712` |
| LB sticky、least-inflight 和计数 | `verl/workers/rollout/llm_server.py:60-143` |
| client acquire/generate/release | `verl/workers/rollout/llm_server.py:170-220` |
| single-turn 一次 generate | `verl/experimental/agent_loop/single_turn_agent_loop.py:38-68` |
| tool Agent 多轮状态机 | `verl/experimental/agent_loop/tool_agent_loop.py:125-285,287-404` |
| vLLM engine drain/pause | `verl/workers/rollout/vllm_rollout/vllm_async_server.py:672-712` |
