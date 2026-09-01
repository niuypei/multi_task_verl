# 多任务推理资源共享调度

本仓库用于设计以独立子仓对接 verl 的多 RL 任务 rollout GPU 动态共享方案，并形成社区 RFC。

## 当前工作入口

- [多RL任务资源共享调度对接VERL - 架构及组件.md](multi_task_scheduler/多RL任务资源共享调度对接VERL%20-%20架构及组件.md)：设计目录顶层的架构与组件工作文档；
- [参考资料索引](multi_task_scheduler/references/README.md)：历次代码调研、架构分析、设计推演和历史归档；
- [长期分析规则](AGENTS.md)：代码证据、版本、实体命名、图示和工作区维护标准。

`multi_task_scheduler/img/` 保存 WIP RFC 直接引用的图片资产；`.obsidian/` 和 `Excalidraw/` 属于本地编辑素材，
不作为正式方案入口。

参考资料只提供证据和设计背景。若参考资料与 WIP RFC、当前代码或用户最新确认结论冲突，以后者为准。
