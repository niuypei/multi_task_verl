# verl 多任务推理资源共享 RFC 工作区

当前方向：在 verl v0.8.0 的 sync + HYBRID 路径上，以独立子仓实现全局 `GroupScheduler`、每任务
`MultiTaskTaskRunner`/`MultiTaskPPOTrainer`/`MultiTaskLLMServerManager`、扩展 LB 和版本化 DDR 权重
存储。初始化使用任务自有资源，运行期支持 rollout 中实时扩缩；动态实例从 DDR 快照加载权重，不要求
训练侧在 rollout 期间重新参与同步。

文档入口：

- [`multi_task_scheduler/00-project-alignment.md`](multi_task_scheduler/00-project-alignment.md)：当前目标、已确认约束、架构和实施顺序；
- [`multi_task_scheduler/01-verl-extension-boundary.md`](multi_task_scheduler/01-verl-extension-boundary.md)：verl v0.8.0 可复用能力、缺口和最小上游扩展面；
- [`multi_task_scheduler/02-group-scheduler-protocol.md`](multi_task_scheduler/02-group-scheduler-protocol.md)：GS↔TaskRunner、LB→GS 协议及模拟器映射；
- [`multi_task_scheduler/03-verl-hybrid-runtime-baseline.md`](multi_task_scheduler/03-verl-hybrid-runtime-baseline.md)：未经改造的 HYBRID 资源启动、组件依赖和单轮迭代代码事实；
- [`multi_task_scheduler/04-multitask-hybrid-runtime-design.md`](multi_task_scheduler/04-multitask-hybrid-runtime-design.md)：多任务改造前后差异、总体扩展方式和 rollout 实时扩缩流程；
- [`multi_task_scheduler/05-multitask-llm-server-manager.md`](multi_task_scheduler/05-multitask-llm-server-manager.md)：`MultiTaskLLMServerManager` 的状态机、事务和接口细化。
- [`multi_task_scheduler/06-taskrunner-direct-control-plane.md`](multi_task_scheduler/06-taskrunner-direct-control-plane.md)：GS 与 TaskRunner 的双向控制面、Actor handle 和并发模型；
- [`multi_task_scheduler/07-rollout-instance-idle-detection.md`](multi_task_scheduler/07-rollout-instance-idle-detection.md)：当前 step 样本完全消耗的判断、LB→GS 空闲上报和原子摘流方案。
- [`multi_task_scheduler/08-versioned-ddr-weight-store.md`](multi_task_scheduler/08-versioned-ddr-weight-store.md)：训练侧发布版本化 DDR 权重快照、动态 replica DDR→HBM 加载及 snapshot 生命周期。
- [`multi_task_scheduler/09-verl-async-standalone-runtime-research.md`](multi_task_scheduler/09-verl-async-standalone-runtime-research.md)：对接 verl fully-async Trainer/Rollouter/MessageQueue 的分离式资源共享方案、异步需求信号、partial rollout、动态生命周期和 DDR 权重数据面；作为 HYBRID-first 之外的备选方案调研。

建议按编号阅读。04 是总体设计，05 是其中任务级本地执行器的下钻设计；两者以 00 的已确认约束为准。
06、07 和 08 分别下钻控制链路、rollout 空闲判定与权重数据面，已经纳入当前方案。09 是备选分离
架构调研，不改变 00 中的当前决策。

历史讨论位于 [`multi_task_scheduler/99-alignment-raw-archive.md`](multi_task_scheduler/99-alignment-raw-archive.md)
和 Git 历史，不代表当前方案。
