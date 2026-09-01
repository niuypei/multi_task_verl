# 多任务资源共享调度参考资料索引

> 状态：参考资料归档。
>
> 整理日期：2026-08-31。

本目录集中保存历次 Agent 产出的代码调研、架构分析和设计推演。正式工作文档仍是上一级目录中的
[多RL任务资源共享调度对接VERL - 架构及组件.md](../多RL任务资源共享调度对接VERL%20-%20架构及组件.md)。本目录内容不自动构成最终 RFC 结论；
发生冲突时，以当前代码、用户最新确认结论和 WIP RFC 为准。

## 当前代码与架构参考

| 文档 | 主题 |
|---|---|
| [00-project-alignment.md](00-project-alignment.md) | 项目背景、目标和已确认约束 |
| [01-redesign-scope.md](01-redesign-scope.md) | 重新设计范围、评审方式和模式边界 |
| [02-group-scheduler-protocol.md](02-group-scheduler-protocol.md) | GroupScheduler 模拟器、控制协议和语义映射 |
| [03-hybrid-standalone-component-topology.md](03-hybrid-standalone-component-topology.md) | HYBRID/STANDALONE 组件、引用、运行实体和部署拓扑 |
| [04-hybrid-initialization-process.md](04-hybrid-initialization-process.md) | HYBRID 初始化和动态创建推理实例 |
| [05-standalone-initialization-process.md](05-standalone-initialization-process.md) | STANDALONE 初始化和动态创建推理实例 |
| [06-standalone-inference-resource-bubble-detection.md](06-standalone-inference-resource-bubble-detection.md) | 异步模式下的空泡、空闲和长尾判断 |
| [07-async-parameter-sync-checkpoint-engine.md](07-async-parameter-sync-checkpoint-engine.md) | Checkpoint Engine effective replicas 与参数同步 |
| [08-forced-reclaim-request-continuation.md](08-forced-reclaim-request-continuation.md) | 强制回收、abort 和 partial rollout 续推 |
| [09-ray-worker-group-actor-binding-mechanism.md](09-ray-worker-group-actor-binding-mechanism.md) | RayWorkerGroup 与业务 Actor 的绑定机制 |
| [10-training-side-worker-group-business-worker-association.md](10-training-side-worker-group-business-worker-association.md) | 训练侧 WorkerDict、包装类和业务 Worker 关系 |
| [11-training-worker-dict-ray-actor-creation-call-chain.md](11-training-worker-dict-ray-actor-creation-call-chain.md) | WorkerDict Ray Actor 完整创建调用链 |
| [12-python-class-instance-call-and-rayclasswithinitargs.md](12-python-class-instance-call-and-rayclasswithinitargs.md) | Python 类/实例调用语义与 RayClassWithInitArgs |
| [14-verl-v0.8-v0.9-hybrid-standalone-differences.md](14-verl-v0.8-v0.9-hybrid-standalone-differences.md) | verl v0.8.0 到 v0.9.0 的架构、组件和流程差异 |
| [15-verl-v0.9-hybrid-colocate-async-partial-rollout-interrupt-resume.md](15-verl-v0.9-hybrid-colocate-async-partial-rollout-interrupt-resume.md) | v0.9 共卡 partial rollout 中断与续推全链 |
| [16-verl-v0.9-dynamic-replica-view-and-reference-audit.md](16-verl-v0.9-dynamic-replica-view-and-reference-audit.md) | HYBRID/STANDALONE 动态 replica 的引用持有者、视图修改矩阵、事务顺序和实现 GAP |
| [17-verl-v0.9-dynamic-replica-scaling-timing-and-mutual-exclusion.md](17-verl-v0.9-dynamic-replica-scaling-timing-and-mutual-exclusion.md) | STANDALONE Fully Async、HYBRID partial、HYBRID 同步三种主链的逐阶段扩缩容时机、示例和互斥协议 |

## 工作记录与早期材料

| 文档 | 主题 |
|---|---|
| [13-project-progress.md](13-project-progress.md) | 当前基线、有效结论、进展和未决问题 |
| [99-alignment-raw-archive.md](99-alignment-raw-archive.md) | 早期需求讨论与原始对齐记录 |
| [backup.md](backup.md) | 旧版内容备份，仅供追溯 |

## 历史归档

- [verl v0.8.0 调研索引](archive/verl-v0.8.0/README.md)：v0.8.0 代码事实、运行模式、Checkpoint Engine、
  STANDALONE 方案和校对产物；
- [HYBRID AS-IS 类图](archive/HYBRID模式AS-IS类图.md)；
- [HYBRID AS-IS 部署视图](archive/HYBRID模式AS-IS部署视图.md)。
