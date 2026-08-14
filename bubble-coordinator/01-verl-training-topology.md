# verl v0.8.0 训练逻辑全貌

> 来源：2026-07-10 端到端代码调研（基线 v0.8.0 tag 7aed6b23）。全部带行号证据。
> 用途：为协调器设计提供"verl 现状地图"——组件关系、部署形态、step 时序、扩展点、空泡位置。

---

## A. 入口与组件总览

### A1. 入口（sync vs async）
- **Sync 入口（v0.8.0 推荐）**：`verl/trainer/main_ppo_sync.py:1843-1866` `main()` → `run_ppo(config, task_runner_class=TaskRunner)`（`main_ppo.py:52`）→ `TaskRunner.run`（`main_ppo_sync.py:1818`）实例化 `PPOTrainer`（`:501`）并调 `trainer.fit()`（`:1836`）。
- **Async/legacy 入口（已 deprecated）**：`verl/trainer/main_ppo.py:35-48`，注释明确"will be replaced by main_ppo_sync.py in v0.8.0"，用 `RayPPOTrainer`（`verl/trainer/ppo/ray_trainer.py:286`）。
- 两者共享 `run_ppo`（`main_ppo.py:52-108`）：`ray.init` + 把 TaskRunner 包成 Ray remote actor（`:82-102`）再 `ray.get(runner.run.remote(config))`。

### A2. 角色/组件
角色定义在 `verl/trainer/ppo/utils.py` 的 `Role`/`WorkerType`（`main_ppo_sync.py:85` import）：
- **Actor + Rollout + Ref（合一）**：`ActorRolloutRefWorker`（`verl/workers/engine_workers.py`），`main_ppo_sync.py:103/1772` 注册。
- **Critic**：`TrainingWorker`（同文件），`:1778`，仅 `need_critic(config)` 为真。
- **RewardModel**：`RewardLoopManager`（`verl/experimental/reward_loop`，`:693`），可选；reward 也可纯函数（无 RM worker）。
- **TeacherModel**：`MultiTeacherModelManager`（`verl/experimental/teacher_loop`，`:702`），仅 distillation 启用。
- 必有：Actor+Rollout（合一于 `ActorRolloutRefWorker`）。可选：Critic、Ref（可融进 actor via LoRA，`:683`）、RewardModel、Teacher。

### A3. Ray 落地形态
- 资源通过 **placement group**。`ResourcePoolManager`（`verl/single_controller/ray/base.py:182`）在 `create_resource_pool`（`:192`）为每个 spec 调 `RayResourcePool.get_placement_groups`（`:130`，用 `ray.util.placement_group`，`:153`），strategy 默认 `STRICT_PACK`（`:130`）。
- 每个角色是 **worker group**（`RayWorkerGroup`，`base.py:424`）非单个 actor；worker group 内每 rank 一个 Ray actor，按 bundle index 调度到 PG（`:524`/`:674`）。Actor/Rollout/Critic 可共池 `global_pool`（`main_ppo_sync.py:1773/1779`），RM/Teacher 可单独 pool（`:1797-1814`）。
- 单个 driver = `TaskRunner`（Ray remote actor，`main_ppo.py:82-83`）。

## B. 一个 step 的时序（纯串行 sync 路径）

sync 主循环 `PPOTrainer.step`（`main_ppo_sync.py:1688`）/ `fit`（`:1589`）。一个 step 调用顺序：
1. **生成采样**：`async_rollout_manager.generate_sequences(batch)`（`:1706`），fire-and-forget dispatch。
2. **拿样本（阻塞）**：`replay_buffer.sample(partition_id="train", ...)`（`:1710`），轮询至所有 prompt `status==success`。
3. **sleep 推理 replica**：`checkpoint_manager.sleep_replicas()`（`:1712`）——腾 GPU 给训练。
4. （可选）colocate reward：`_compute_reward_colocate`（`:1717`，当前 `NotImplementedError`）。
5. balance batch（`:1723`）→ 6. old log prob（`:1727`，actor 推理）→ 7. （可选）ref log prob（`:1732`）→ 8. （可选）critic values（`:1737`）→ 9. advantage（`:1741`，driver 算 GRPO）→ 10. （可选）update critic（`:1746`）→ 11. **update actor（`:1751`，反向+优化器）**。
12. 回 fit（`:1640`）：可选 save ckpt → **`checkpoint_manager.update_weights()`（`:1651`，trainer→rollout 推权重 + wake up）** → validate → metrics → cleanup TQ。

### B5. 空泡物理位置（项目核心）
- **训练卡闲着等推理**：step 内 `gen`（`:1709` marked_timer）到 `sample` 返回（`:1710`）+ `sleep_replicas`（`:1712`）之间——训练卡空闲等 rollout。
- **推理卡闲着等训练（最大推理空闲段）**：`sleep_replicas`（`:1712`）之后直到 `update_weights`（fit `:1651`）——replica 被 sleep/挂起，训练卡跑 old_log_prob/ref/values/critic/actor。这是同步路径**最大的推理空闲段**，也是协调器要填的空泡主战场。

### B6. 权重同步路径
- 走 **checkpoint engine**（默认 backend 是 naive）。`CheckpointEngineManager.update_weights`（`verl/checkpoint_engine/base.py:470`）：backend=="naive"（`:478-480`）进程内直传；否则（nccl/nixl/hccl/kimi/mooncake）abort replicas → release kv_cache → build process group → 跨进程同步。
- 触发点：sync 每 step 末 fit `:1651`；init 一次 `:1606`。默认配置 `verl/trainer/config/rollout/rollout.yaml:264-270` `backend: naive`。现成 sync+hybrid 走 naive 进程内。

## C. 部署形态：共卡 vs 分离

### C7. sync 路径强制 hybrid
- `PPOTrainer.init_workers` 调 `LLMServerManager.create(config, worker_group=self.actor_rollout_wg, ...)`（`main_ppo_sync.py:712-714`），**硬传 worker_group**。
- `LLMServerManager.__init__` 断言 `worker_group is not None or nnodes>0`（`llm_server.py:248`）；`_initialize_llm_servers`（`:314-325`）：worker_group 非空 → `init_hybrid`（HYBRID，`replica.py:137`），否则 → `init_standalone`（`:325`）。
- async 还有 `assert self.hybrid_engine`（`ray_trainer.py:334`），config 默认 `hybrid_engine: true`（`ppo_trainer.yaml:53`）。
- `RolloutMode` 三态（`replica.py:54-67`）：HYBRID（共进程，需权重同步）/ COLOCATED（同 PG 不同进程，免权重同步，GRM）/ STANDALONE（独立 GPU，disaggregated，off-policy）。

### C8. sync 走 STANDALONE 的最小改动点
- 改 `main_ppo_sync.py:712-714`：`worker_group=self.actor_rollout_wg` → `worker_group=None`。后果：`llm_server.py:248` 断言要 `rollout.nnodes>0`（`rollout.yaml:11` 从 0 改 >0），`llm_server.py:297-301` 用 `n_gpus_per_node*nnodes` 算 world_size → 要 rollout 独立 GPU 规格。然后走 `init_standalone`（`:325`）。
- `actor_rollout_wg` 仍是合体 actor（不再带 rollout），rollout 由独立 replica 持有；`checkpoint_manager`（`:732-736`）`backend=naive`（`rollout.yaml:270`）standalone 下失效（naive 要求进程内 colocate），需改 `backend: nccl/nixl`。
- **最小改动 = `:712` worker_group=None + rollout.yaml(nnodes/独立GPU) + checkpoint_engine backend 非 naive**。

### C9. experimental 分离模块（都 off-policy/async）
- `verl/experimental/separation/`：`SeparateRayPPOTrainer`（`ray_trainer.py:53`）— 抽象基类，拆 init_workers 为 `_init_resource_pools/_create_worker_classes/_init_worker_groups/_init_models/_init_async_rollout_manager`（`:113-118`）。one_step_off_policy 与 fully_async 的共同基类。
- `verl/experimental/one_step_off_policy/`：`OneStepOffRayTrainer`（`ray_trainer.py:50`），`assert not self.hybrid_engine`（`:89`），`_init_async_rollout_manager` 调 `LLMServerManager.create(config=self.config)`（`:190`，不传 worker_group → STANDALONE）。**off-policy**，不可直接用于 on-policy sync。
- `verl/experimental/fully_async_policy/`：`FullyAsyncTrainer`+`FullyAsyncRollouter`+`FullyAsyncLLMServerManager`（`:153`，支持 hybrid+standalone 混在）。入口 `fully_async_main.py:212`。**完全异步流式，off-policy**，不可直接用于 on-policy sync。
- **结论：现成分离模块全是 off-policy/async，没有现成"on-policy sync + STANDALONE"组合。**

## D. 多任务与资源

### D10. 多任务能力
- 单 Ray cluster 可跑多任务，但 verl **天然单任务**：`run_ppo`（`main_ppo.py:52-108`）一次只起一个 TaskRunner remote actor 并 `ray.get` 阻塞等它结束（`:102`）。无 task/job 抽象、无多 driver 调度器。多任务只能靠用户自起多 driver 进程各自 `ray.init`（共享同一 cluster）或外部编排。config 层无多任务字段。

### D11. 资源规格 + 运行时改 rollout GPU 数
- 规格：`trainer.nnodes/n_gpus_per_node`（`ppo_trainer.yaml:143-146`）定义 `global_pool`（`main_ppo.py:161-164`/`main_ppo_sync.py:1786-1787`）。rollout 细节 `rollout.tensor_model_parallel_size/data_parallel_size/pipeline_model_parallel_size`（`rollout.yaml:56-67`）、`rollout.nnodes/n_gpus_per_node`（`rollout.yaml:11/14`，standalone 用）。
- 运行时改 rollout GPU 数：sync `PPOTrainer` **不支持**。`CheckpointEngineManager` 有 `add_replicas/remove_replicas`（`base.py:414-429`）接口，但 sync 路径无调用方。只有 `FullyAsyncLLMServerManager`（`fully_async_rollouter.py:228/290`）实现并被暴露调用。**sync 路径运行时改 GPU 数只能改源码。**

### D12. 异常退出资源回收
- verl **无显式资源回收路径**。`TaskRunner.run` 的 `finally`（`main_ppo_sync.py:1837-1840`）只关 `replay_buffer`/`tq.close()`，**不 kill actor / 不 remove PG**。
- 依赖 Ray 自动回收：actor 异常 → Ray 标记 dead；PG lifetime 随创建它的 driver actor，driver 退出则 PG 回收（Ray 默认，非 verl 显式）。无 `ray.kill`/`remove_placement_group` 调用点。**崩溃后 PG 残留风险看 Ray GC，verl 不兜底。**

## E. 可观测与扩展点

### E13. phase 暴露机制
- **无对外 phase 状态**。phase 仅 driver 进程内 `marked_timer`（`main_ppo_sync.py:1098/1639/1709/1726/...`）记入 `timing_raw`，step 末经 `compute_timing_metrics`（`ray_trainer.py:1731`）→ `logger.log`（`:1676`）打 wandb/console（**事后**，非实时）。
- 外部进程不改源码能拿到的：step 级 timing 指标（事后）。无实时 RPC 查"当前在 gen/training"。TransferQueue `kv_list`（`:215`）可查样本状态，但是数据状态非 phase。**现成 sync 路径不能让外部实时知道"rollout 闲"。**

### E14. 扩展点（FQN 类替换，非 callback）
- **无通用 callback/hook/plugin 体系**。
- 5 个 FQN 类替换扩展点（config 填 FQN → 运行时反射加载子类）：
  | 扩展点 | 位置 | 接入方式 |
  |-------|------|---------|
  | `task_runner_class` | `main_ppo.py:52/82`、`main_ppo_sync.py:1862` | 换 TaskRunner 子类 |
  | `actor_rollout_ref.rollout.agent.agent_loop_manager_class` | `main_ppo_sync.py:716-720` | 换 AgentLoopManager 子类 |
  | `actor_rollout_ref.rollout.checkpoint_manager_class` | `ray_trainer.py:960-964` | 换 CheckpointEngineManager 子类 |
  | `checkpoint_engine.custom_backend_module` | `base.py:287-303`、`rollout.yaml:290-292` | import 外部模块注册新 backend |
  | `rollout.name` → `RolloutReplicaRegistry` | `replica.py:302-310` | 换 replica 实现 |
- 要 hook step 各阶段，只能继承 `PPOTrainer`/`RayPPOTrainer` 覆写 `step`/`fit_step`，无固定 hook 点。

### E15. 外部进程触发扩缩容/权重重同步
- sync 路径：扩缩容和权重重同步**只能在 driver 进程内**。`checkpoint_manager.update_weights`（`base.py:470`）是 driver 进程方法调用（`main_ppo_sync.py:1651`）；`add_replicas/remove_replicas`（`base.py:414/422`）sync 路径无调用方。
- **sync 路径 `PPOTrainer` 是 driver 进程内普通对象（非 Ray actor，`main_ppo_sync.py:1830-1834`），外部无法 RPC 它。**
- 外部触发的现成能力只在 fully_async：`FullyAsyncRollouter` 是 Ray remote actor（`fully_async_main.py:119`），`add_replicas/remove_replicas`（`fully_async_rollouter.py:1134-1138`）可 `.remote()` 调；`FullyAsyncTrainer` 也是 remote（`:146`），`_fit_update_weights` 可 `.remote()`（`:110`）。**但仅 fully_async 路径**。结论：sync 路径无外部触发接口，须改源码或自建外部编排层。

## F. 对协调器项目的关键影响（提炼）

1. **集成面现实**：verl 无 callback，但有 5 个 FQN 类替换扩展点 → 外部插件可"写子类+config挂载"接入，不必直接改 verl 源码。这是 P1/P2 的现实落脚点。
2. **sync+STANDALONE 现成走不通**（两道硬卡 `main_ppo_sync.py:712` 传 worker_group + `ray_trainer.py:334` assert hybrid；naive backend 不支持跨进程），但分离地基在（`init_standalone`、`CheckpointEngineManager.add/remove_replicas`）。
3. **现成分离模块都绑 off-policy/async**（one_step_off_policy / fully_async），不可直接用于 on-policy sync；但 fully_async 的 remote actor add/remove_replicas 是协调器想要的外部可控能力，只是绑 async 路径。
4. **sync 任务对外是黑盒**：PPOTrainer 非 Ray actor，外部无法 RPC；phase 无实时暴露；资源回收无 verl 兜底。协调器要纳管多 sync 任务须自建：(a) 多 driver 编排层，(b) phase 探测（verl 无现成），(c) 资源回收自兜底。
5. **空泡主战场**：`sleep_replicas`→`update_weights` 之间是最大推理空闲段（replica 挂起、训练卡跑反向），这是协调器要填的空泡核心位置。
