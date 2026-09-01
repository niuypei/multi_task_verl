# multi_task_verl Repository Instructions

本文件是本仓库的长期工作记忆。进入本仓库后，任何 verl 架构调研、RFC 分析、文档修改或实现工作开始前，
必须先阅读本文件，并读取 `multi_task_scheduler/references/13-project-progress.md` 获取当前阶段状态。

如果用户后续明确修改了某项约束，应同步更新本文件和进展文档；较新的用户确认优先于本文件中的旧结论。

## 1. 分析前强制检查

每次开始分析前依次执行：

1. 阅读本文件和 `multi_task_scheduler/references/13-project-progress.md`。
2. 判断请求属于事实解释、代码分析、方案审视、文档更新还是代码实现，不扩大用户授权范围。
3. 检查 `/Users/nyp/Documents/verl` 的实际 `HEAD`、tag/describe 和工作区状态，不根据历史会话假定版本。
4. 检查 `/Users/nyp/Documents/multi_task_verl` 的工作区状态，保留用户已有修改。
5. 明确本次结论属于 AS-IS、TO-BE 还是 GAP。
6. 先形成代码证据链，再得出结论；用户提出的设计思路必须先审视正确性和约束，经确认后再写入正式方案。
7. 如果代码基线、用户已确认方向或既有文档互相冲突，先指出冲突，不静默选择其中一个。

## 2. 项目背景与长期目标

- 项目目标是以独立子仓方式对接 verl，实现多 RL 任务之间的 rollout GPU 动态共享，并形成可提交给社区的 RFC。
- 调度项目仓库：`/Users/nyp/Documents/multi_task_verl`。
- GroupScheduler 模拟器：`/Users/nyp/Documents/multi-rl-task-scheduler`。
- verl 源码：`/Users/nyp/Documents/verl`。
- 主要新增组件应实现于独立子仓，尽量减少对 verl 的侵入。
- 对 verl 的必要侵入应局限在原生扩展点无法覆盖的生命周期能力，例如训练运行期间动态创建、挂接、回收推理实例。
- 当前部署分析主轴只包括 `RolloutMode.HYBRID` 和 `RolloutMode.STANDALONE`。只有在解释代码边界时才讨论
  `RolloutMode.COLOCATED`，不得把它提升为本 RFC 的第三种主部署模式。
- `trainer.v1.trainer_mode=colocate_async` 是 Trainer 运行模式，不等同于 `RolloutMode.COLOCATED`；当前代码路径仍可使用
  HYBRID rollout 资源组织，分析时必须分别说明 Trainer mode 和 RolloutMode。

## 3. 已确认的 TO-BE 架构约束

### 3.1 GroupScheduler、TaskRunner 与任务内组件

- `GroupScheduler` 是全局调度大脑，与多个 `TaskRunner` 形成一对多关系。
- `GroupScheduler` 和 `TaskRunner` 互相持有 Ray ActorHandle。
- 任务启动时向 `GroupScheduler` 注册自身资源；任务结束时注销资源。
- `GroupScheduler` 通过心跳维护任务、节点、GPU 和资源变化。
- `GroupScheduler` 通过目标任务的 `TaskRunner` 下发创建、扩容、回收、sleep、wake up 等命令。
- 不为该控制链额外引入通信 Actor 或中转组件。
- `TaskRunner` 必须持有或能够访问 `self.group_scheduler`，以完成注册、注销和状态上报。
- `MultiTaskLLMServerManager` 是任务 controller 进程中的普通对象，不是 Ray Actor；跨任务控制必须经
  `TaskRunner` 落到该对象。
- 组件名称使用 `MultiTaskLLMServerManager`，不使用旧名称 `ManagedLLMServerManager`。
- `GlobalRequestLoadBalancer` 持有 `GroupScheduler` ActorHandle，将本任务已确认空闲的推理实例上报给
  `GroupScheduler`，但不负责全局调度决策。

### 3.2 初始化和实时伸缩

- 初始 replica 数量仍由每个任务自身配置决定，`GroupScheduler` 不分配任务初始化规模。
- 初始化阶段任务只注册资源；资源共享和实时扩缩发生在训练运行阶段。
- 基础推理实例规模按已确认场景大于 1，不以单实例作为架构前提。
- donor 原生推理实例不销毁，只执行 abort/drain、从活跃视图移除和 sleep，便于随时唤醒。
- borrower 的临时推理实例可以在回收时销毁或按后续设计处理，但不能错误地销毁 donor 原生实例。
- 动态 borrower 实例不能默认复用 donor 的 `RayResourcePool`、Placement Group 或 resource bundle。
- borrowed replica 仅根据 `GroupScheduler` 租约中的 immutable node ID/GPU IDs 创建；不把 donor 的
  ResourcePool、PG、bundle、`RayWorkerGroup`、worker、`CheckpointEngineWorker`、`ServerAdapter` 或 server handle
  传给 borrower。
- borrowed replica 从创建开始完全归 borrower 任务所有。borrower 的 `MultiTaskLLMServerManager` 持有完整运行时和
  生命周期引用；borrower 的同步组件和 LB 分别持有自己的同步 endpoint 与 head server 引用；donor 只管理自己的
  native sleeping replica，不登记 borrowed replica。
- 如果绕过 verl 原生资源分配视图并按 node ID、GPU ID、NodeAffinity 和显式可见设备创建实例，真实 GPU 排他性和
  生命周期必须由 `GroupScheduler` 维护。

### 3.3 参数同步

- `GroupScheduler` 不改变 verl 原生参数同步时机。
- `GroupScheduler` 不持有 replica/worker/adapter runtime handles，也不直接调用参数同步组件；它通过目标任务的
  `TaskRunner` 下发规模命令，由 borrower 任务更新参数同步组件当前感知的有效 replica 集合。
- 有效集合应表达本任务固有 replica、当前受赠 replica、已捐出 replica 和正在回收 replica 的状态变化。
- 新实例必须完成明确版本的权重同步后才能进入 LB 接收请求。
- 不得假定原生 `CheckpointEngineManager` 自动覆盖 LB 中所有 server；必须从其实际 `replicas`、
  `replica.workers` 和 ActorHandle 调用路径证明覆盖范围。
- 不得虚构 verl 已有 `BorrowedCheckpointEngineWorker` 类。TO-BE 新类必须明确标注为 Proposed，并说明创建方式。
- borrowed replica 的同步 receiver/`ServerAdapter` endpoint 必须由 borrower 创建并拥有；不得通过重绑定或临时使用
  donor `CheckpointEngineWorker`/`ServerAdapter` 解决。
- 当前阶段优先分析 Checkpoint Engine 固有能力；除非用户重新要求，否则不要默认依赖 Mooncake/DDR offload 解决同步。

### 3.4 partial rollout 与回收

- partial rollout 是生成阶段的中断和续推机制，不等于使用半条 trajectory 进行 PPO 更新。
- 训练样本是否完整必须依据 ReplayBuffer、TransferQueue 和 AgentLoop 的真实状态流判断。
- 强制回收时，应优先审视能否复用 verl 原生 abort/resume、token 前缀保存、剩余 token budget 和版本记录机制。
- 必须明确 partial 状态保存在服务端、TransferQueue、客户端协程还是其他组件；不能笼统写“请求被保存”。
- 跨 replica 迁移、进程故障和 checkpoint 恢复不能从普通 abort/retry 行为直接推导，必须单独证明。

## 4. 代码分析原则

### 4.1 一切以代码为准

- 文档、注释、类名和变量名只能作为线索，最终结论必须由实际分支、对象创建和调用路径支持。
- 代码注释与实现不一致时，以实现为准，并明确指出注释偏差。
- 不只搜索类定义；必须继续追踪谁构造它、谁持有返回值、谁调用方法以及方法最终在哪个进程执行。
- 分析每个关键调用时都说明主语、宾语、输入、返回值和副作用。

### 4.2 版本和行号

- 每份代码分析文档必须记录源码目录、tag/describe、commit 和分析日期。
- v0.8.0 文档只能作为历史资料，不能作为 v0.9.0 当前实现证据。
- 所有关键实现结论标注 `文件:行号`；版本升级后必须重新校验行号和分支。
- 如果当前 checkout 不是精确 tag，必须写出完整 describe，例如 `v0.9.0-1-g88512193`，不能简称为精确
  `v0.9.0`。

### 4.3 AS-IS、TO-BE 和 GAP

- AS-IS 只使用 verl 当前真实存在的类、函数和流程。
- TO-BE 中新增组件或类必须标注 Proposed/扩展类，不能伪装成 verl 现有实体。
- GAP 必须说明哪个 AS-IS 能力无法满足哪个 TO-BE 需求，以及对应证据。
- 用户要求先识别边界和扩展点时，不提前扩展为完整设计或实现计划。
- 用户要求分析但未要求修改文档时，只输出分析，不擅自落盘。

### 4.4 Python 与 Ray 对象判定

判断某个类是不是实际 Ray Actor，必须找到完整证据链：

```text
原始 Python 类
→ ray.remote(Class) 或 @ray.remote
→ ActorClass
→ ActorClass.options(...).remote(...) 或 ActorClass.remote(...)
→ ActorHandle
```

- `ray.remote(Class)` 只创建 ActorClass 描述，不创建 Actor 实例。
- 只有 ActorClass 的 `.remote()` 才创建 Ray Actor 并返回 ActorHandle。
- `SomeClass(...)` 是类对象调用，由元类 `type.__call__()` 驱动 `__new__()` 和 `__init__()`。
- `some_instance(...)` 才调用 `type(some_instance).__call__()`。
- `RayClassWithInitArgs(...)` 创建包装实例；对该实例执行 `cia(...)` 才进入
  `RayClassWithInitArgs.__call__()`。
- 必须区分 `RayWorkerGroup` 普通代理对象、其 `_workers` 中的 ActorHandle、远端 Ray Actor，以及 Actor 进程内普通对象。

### 4.5 调用链完整性

至少追踪以下阶段：

```text
入口配置
→ 类选择/注册
→ 构造参数包装
→ ResourcePool/PG/bundle
→ ActorClass 创建
→ Actor 实例创建
→ ActorHandle 保存
→ 角色代理生成
→ 远端方法调用
→ 业务对象实际执行
```

对于方法代理，继续追踪：

```text
@register 元数据
→ 动态方法绑定
→ dispatch/collect/blocking
→ ActorHandle.method.remote()
→ Actor 内委托对象
```

## 5. 实体和图示标准

- AS-IS 图中的实体名称必须使用代码中真实存在的类名、函数名或明确的运行时对象名称。
- 禁止使用不存在的 `TrainingActor`、任意 `X_` 前缀或无代码依据的占位类。
- TO-BE 实体必须明确标注 Proposed，避免与 AS-IS 混淆。
- 每个实体标注运行类型：本地/driver 进程、Ray Actor、ActorHandle、controller 普通对象、Actor 内普通对象、
  vLLM 子进程或外部存储。
- 类引用、对象持有、远程调用、数据流和部署关系不能混在一张没有图例的图中。
- 每张图必须附文字解释，并说明关键实体的代码文件和行号。
- Mermaid 图必须保证语法可渲染；flowchart 不使用 classDiagram 专属组合箭头。
- 类图只表达类和持有关系；部署图表达进程、节点、GPU 和 Actor；时序图表达调用先后和返回值。
- 图中的调用应尽量使用真实方法名，文字描述必须具有明确主语和宾语。

## 6. 三种视图的分析标准

任何资源共享方案都必须分别分析：

1. verl/Ray 原生资源视图：`RayResourcePool`、Placement Group、bundle、`RayWorkerGroup`。
2. 推理 LB 视图：server address、`vLLMHttpServer` ActorHandle、路由缓存和请求负载。
3. `GroupScheduler` 全局物理视图：task、replica、node ID、GPU ID、donor/borrower 和生命周期状态。

除此之外，动态 replica 必须单独分析 borrower 任务内的 manager ownership/lifecycle registry 和 Checkpoint Engine
同步投影。LB、同步投影与 manager registry 不能互相替代或自动推导。

如果三种视图不一致，必须说明：

- 哪个组件是事实来源；
- 谁保证 GPU 排他性；
- 创建、加入 LB、权重同步、回收和故障时如何原子更新；
- 失败后如何回滚或隔离；
- 原生调度器为什么不会再次错误分配对应资源。

## 7. 空泡和空闲判定标准

- 不得仅根据瞬时并发数为 0 判断可捐赠。
- 同步模式需证明本 step 所有待调度 prompt 已经提交，上游没有未发送样本，且当前实例无 queued/running 请求。
- 异步模式需同时分析 inflight window、TransferQueue 状态、补充 prompt 的条件、陈旧度阈值、drop/wait 策略和
  backpressure。
- 没有陈旧度或并发窗口控制的 fully async 流水线可能持续产生请求，不能自然推导出可捐赠空泡。
- 必须给出判断公式、伪码和至少一个数值示例，并说明误判风险。

## 8. 文档维护标准

- `multi_task_scheduler/references/` 统一存放 RFC 的分析与调研参考资料；正式工作文档保留为
  `multi_task_scheduler/【WIP】多RL任务资源共享调度RFC.md`。
- `multi_task_scheduler/references/archive/verl-v0.8.0` 是历史归档，除非用户明确要求，不修改其事实基线。
- v0.9.0 新分析优先新建独立文档，评审通过后再合并到 RFC。
- 每份文档开头标明状态，如“待评审”“已确认”“历史归档”。
- 引用其他文档时说明引用的是历史事实、当前事实还是设计背景。
- 新结论与已有文档冲突时，全量检索并修正所有受影响图、表、文字和代码行号。
- 完成一个实质阶段后更新 `multi_task_scheduler/references/13-project-progress.md`，不要把阶段性进度写进本长期规则文件。

## 9. Git 和工作区标准

- 工作区可能包含用户未提交改动；不得覆盖、清理或顺带格式化无关文件。
- 提交前显式列出目标文件，只暂存与当前任务有关的内容。
- `.obsidian/`、插件、主题、临时 Excalidraw 草稿和“未命名”笔记默认不提交，除非用户明确要求。
- 提交前执行 whitespace 检查、查看 staged diff/stat，并核对删除和二进制图是否符合预期。
- 不因“推送代码”自动把所有 untracked 文件加入仓库。
- 文档更新与源码实现使用能准确描述内容的提交信息。

## 10. 分析交付自检清单

在结束一次分析前确认：

- [ ] 已核对实际 verl commit/tag。
- [ ] 已说明结论属于 HYBRID、STANDALONE、Trainer mode 还是其他维度。
- [ ] 已区分 Ray Actor、ActorClass、ActorHandle、普通对象和子进程。
- [ ] 已给出关键类、函数、调用点和返回值的代码行号。
- [ ] 已从入口追到最终执行实体，没有停在中间包装层。
- [ ] 已区分 AS-IS、TO-BE 和 GAP。
- [ ] 已审视用户方案的正确性和限制，而不是直接接受。
- [ ] 图中实体名真实、类型明确、Mermaid 可渲染并有文字解释。
- [ ] 未修改用户未授权的文件。
- [ ] 如有实质进展，已更新进展文档。
