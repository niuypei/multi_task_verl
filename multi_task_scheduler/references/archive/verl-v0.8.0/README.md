# verl v0.8.0 调研与设计归档

> 归档基线：`/Users/nyp/Documents/verl`，tag `v0.8.0`，commit `7aed6b23`。
> 本目录保存已经完成的 verl 代码调研、对接设计和评估产物。内容用于追溯，不自动代表当前方案。

主目录继续维护的需求对齐与模拟器分析：

- [项目背景与需求对齐](../../00-project-alignment.md)
- [GroupScheduler 控制协议与模拟器映射](../../02-group-scheduler-protocol.md)
- [早期需求讨论原始留痕](../../99-alignment-raw-archive.md)

## 1. verl 代码基线与扩展边界

- [01-verl-extension-boundary.md](01-verl-extension-boundary.md)：可复用接口、实现缺口和最小上游扩展面；
- [03-verl-hybrid-runtime-baseline.md](03-verl-hybrid-runtime-baseline.md)：HYBRID 资源启动、进程/Actor 拓扑和单轮迭代代码事实；
- [16-verl-hybrid-vs-colocated-runtime.md](16-verl-hybrid-vs-colocated-runtime.md)：HYBRID 与 COLOCATED 的组件、引用、调用和权重生命周期差异；
- [17-verl-checkpoint-engine-runtime.md](17-verl-checkpoint-engine-runtime.md)：Checkpoint Engine 控制面、两跳数据面、backend 和使用场景。

## 2. 多任务 HYBRID 扩展设计

- [04-multitask-hybrid-runtime-design.md](04-multitask-hybrid-runtime-design.md)：总体扩展设计；
- [05-multitask-llm-server-manager.md](05-multitask-llm-server-manager.md)：任务级推理实例执行器；
- [06-taskrunner-direct-control-plane.md](06-taskrunner-direct-control-plane.md)：GroupScheduler 与 TaskRunner 双向控制面；
- [07-rollout-instance-idle-detection.md](07-rollout-instance-idle-detection.md)：step 样本耗尽和空闲实例识别；
- [08-versioned-ddr-weight-store.md](08-versioned-ddr-weight-store.md)：版本化 DDR 权重快照和动态实例加载。

## 3. 训推分离与异步模式

- [09-verl-async-standalone-runtime-research.md](09-verl-async-standalone-runtime-research.md)：分离式多任务推理资源共享分析；
- [10-verl-separated-async-mode-overview.md](10-verl-separated-async-mode-overview.md)：异步模式和共同运行底座总览；
- [11-verl-separated-one-step-off-policy.md](11-verl-separated-one-step-off-policy.md)：One-Step Off-Policy；
- [12-verl-fully-async-mode1-on-policy-pipeline.md](12-verl-fully-async-mode1-on-policy-pipeline.md)：FullyAsync Mode 1；
- [13-verl-fully-async-mode2-stream-off-policy.md](13-verl-fully-async-mode2-stream-off-policy.md)：FullyAsync Mode 2；
- [14-verl-fully-async-mode3-stale-samples.md](14-verl-fully-async-mode3-stale-samples.md)：FullyAsync Mode 3；
- [15-verl-fully-async-mode4-partial-rollout.md](15-verl-fully-async-mode4-partial-rollout.md)：FullyAsync Mode 4。

## 4. 审视与方案评估产物

- [调度文档代码核验](review-artifacts/doc-review.html)
- [方案正确性与可行性分析](review-artifacts/scheme-analysis.html)
