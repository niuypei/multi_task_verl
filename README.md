# 多任务推理资源共享工作区

当前主目录保留需求背景、已确认约束、GroupScheduler 协议，以及当前设计直接依赖的 verl v0.8.0
HYBRID/STANDALONE 拓扑和 HYBRID 动态扩容代码边界。其余历史运行时调研与方案评估统一归档。

## 当前保留文档

- [`multi_task_scheduler/STANDALONE.md`](multi_task_scheduler/STANDALONE.md)：STANDALONE 多任务 rollout 资源共享总体方案，汇总逻辑/部署/运行视图、空泡识别、受赠创建、强制回收续推和 CE effective replicas 参数同步；
- [`multi_task_scheduler/STANDALONE-CODE-AUDIT.md`](multi_task_scheduler/STANDALONE-CODE-AUDIT.md)：逐图、逐表、逐流程核对 STANDALONE 方案与 verl v0.8.0、GroupScheduler 模拟器代码，区分已有事实、模拟能力、可复用锚点和待实现设计；
- [`multi_task_scheduler/00-project-alignment.md`](multi_task_scheduler/00-project-alignment.md)：项目目标、需求边界、已确认约束和目标架构；
- [`multi_task_scheduler/01-redesign-scope.md`](multi_task_scheduler/01-redesign-scope.md)：当前重新设计的评审流程、目标模式范围和 HYBRID/STANDALONE 术语边界；
- [`multi_task_scheduler/02-group-scheduler-protocol.md`](multi_task_scheduler/02-group-scheduler-protocol.md)：`/Users/nyp/Documents/multi-rl-task-scheduler` 两套 GroupScheduler 实现、协议骨架及控制语义映射；
- [`multi_task_scheduler/03-hybrid-standalone-component-topology.md`](multi_task_scheduler/03-hybrid-standalone-component-topology.md)：HYBRID/STANDALONE 类图、引用关系、进程/Ray Actor/普通对象实体和 GPU/Placement Group 部署拓扑；
- [`multi_task_scheduler/04-hybrid-initialization-process.md`](multi_task_scheduler/04-hybrid-initialization-process.md)：HYBRID 初始化过程，以及 donor replica 摘流/sleep 后在同一物理卡创建 receiver vLLM replica、加入 receiver LB、销毁归还 slot 的完整事务；
- [`multi_task_scheduler/05-standalone-initialization-process.md`](multi_task_scheduler/05-standalone-initialization-process.md)：STANDALONE 的 One-Step/Fully Async 初始化链，以及保留 donor ResourcePool/PG/bundle、基于 sleeping replica placement 动态创建 borrower server 的完整扩容事务；
- [`multi_task_scheduler/06-standalone-inference-resource-bubble-detection.md`](multi_task_scheduler/06-standalone-inference-resource-bubble-detection.md)：STANDALONE 的 One-Step 与 FullyAsync 四种异步模式下，replica 瞬时空闲、自然空泡、可抢占容量和可分配 HBM slot 的分层识别方案；
- [`multi_task_scheduler/07-async-parameter-sync-checkpoint-engine.md`](multi_task_scheduler/07-async-parameter-sync-checkpoint-engine.md)：GroupScheduler 动态调整 CE effective replicas、固有/捐赠/受赠实例的统一参数同步、borrower-owned CE endpoint，以及保持 verl 原生同步时机的完整方案；
- [`multi_task_scheduler/08-forced-reclaim-request-continuation.md`](multi_task_scheduler/08-forced-reclaim-request-continuation.md)：异步 rollout 中 GS 强制回收有 in-flight 请求的 replica 时，复用 abort/partial continuation、LB 两阶段摘流和跨实例同版本续推的方案；
- [`multi_task_scheduler/99-alignment-raw-archive.md`](multi_task_scheduler/99-alignment-raw-archive.md)：早期需求背景与约束讨论的原始留痕。

## verl 调研归档

- [`multi_task_scheduler/archive/verl-v0.8.0/README.md`](multi_task_scheduler/archive/verl-v0.8.0/README.md)：verl v0.8.0 代码基线、HYBRID/COLOCATED、训推分离异步、Checkpoint Engine、扩展设计及评估产物索引。

归档内容保留当时的代码事实、设计假设和交叉引用，供后续追溯；除非重新评审，不代表新的当前方案。
