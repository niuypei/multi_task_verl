# Bubble Coordinator — 背景与约束对齐

> 单一事实来源。重组版（2026-07-10）。流水原文已归档至 `00-alignment.raw-archive.md`。
> 配套记忆：`memory/rl-inference-bubble-coordinator.md`、`memory/verl-architecture-non-negotiable.md`
> 状态图例：`[ ]` 待对齐 · `[~]` 有倾向待确认 · `[x]` 已确认

---

## 1. 项目定义

开发一个**独立组件**，搭在 verl 之上，同时纳管多个训推同步 RL 任务：识别任务中的空闲推理实例、回收空闲推理卡到公共池、按需重新分配给繁忙任务、按负载对 RL 任务推理侧扩缩容，降低端到端训练耗时。

## 2. 硬原则与软约束

### 硬原则（用户拍板，不可违反）
- **P1**：不改 verl 现有架构。任何方案若需修改 verl 核心代码，须显式标红并退回重设计。
  - 注：P1 精确边界暂不定，**先调研再定界**（见 §5.2）。
- **P2**：独立组件 / 子项目，插件式接入 verl。集成面 = verl 现有扩展点（config / hooks / 外部进程对 vLLM/Ray 资源的控制）。
- **P3**：先对齐背景与约束，再谈技术路线；讨论全程留痕（本文档即载体）。

### 软约束（路径选择标准）
- **社区易接受**：打通分离的代码路径选择，倾向 复用 verl 已有 experimental/主线能力 > 外部自建旁路 > 改源码 fork。

## 3. 已确认约束（技术栈与机制）

| 项 | 结论 | 证据 / 备注 |
|----|------|------------|
| verl 版本 | **0.8.0 精确 tag**（commit 7aed6b23，detached HEAD） | `verl/version/version`=0.8.0.dev（tag 后下 commit 才 bump，纯字符串差异）；基线已切到此 |
| 部署底座 | **Ray** | 协调器集成面=Ray（actor/PG/Ray Job） |
| 训练侧 | **Megatron（MCore）** | 与 vLLM rollout 的权重同步是关键耦合点 |
| 推理侧 | **vLLM** | RolloutMode 三态见 §4.1 |
| 算法 | **GRPO（on-policy）** | on-policy 约束 → 借调/归还卡须重载当前 step 权重，否则 off-policy（PoC 正确性红线） |
| 训推机制 | **纯串行同步（训推串行）** | generation→weight-sync→training→weight-sync→下一轮；两侧互为空泡 |
| vLLM 在线改 TP | 不可行 | 借调倾向走"临时多开 rollout replica / DP 扩容"，伴随 init+load+warmup 冷启动 |

## 4. 部署形态与 RolloutMode

### 4.1 verl 0.8.0 的三种 RolloutMode（`replica.py:55-67`）
| 模式 | 形态 | verl 注释适用场景 |
|------|------|----------------|
| HYBRID | 训练引擎与 rollout 引擎融合同进程、共享 GPU、靠权重同步切换 | on-policy |
| COLOCATED | rollout 与 hybrid 引擎同 PG 但分进程、共享 GPU、无需权重切换 | GRM(judge) |
| STANDALONE | 独立 rollout server、独立 GPU、disaggregated | off-policy |

权重同步印证（`engine_workers.py:670-696`）：colocated→"naive" 进程内直接同步；disaggregated→checkpoint engine 异步传权重。

### 4.2 目标部署组合与代码现状
- **目标组合**：GRPO + on-policy + 训推串行(sync) + **分离部署(STANDALONE)**。
- **代码结论（硬证据）**：v0.8.0 现成代码**不支持**此组合——
  - sync trainer 创建 rollout server 时硬传 worker_group：`main_ppo_sync.py:712-713`。
  - `LLMServerManager` 分流（`llm_server.py:314-325`）：worker_group 非 None→`init_hybrid`（共卡）；仅 None 且 nnodes>0→`init_standalone`（分离）。故 sync trainer 永走 hybrid。
  - 硬断言 `ray_trainer.py:333-334` `assert self.hybrid_engine`。
  - docstring `main_ppo_sync.py:15`："Synchronous PPO trainer with colocated actor and rollout"。
- **但分离地基存在**：`RolloutMode.STANDALONE` / `init_standalone()` / `LLMServerManager` 的 standalone 分支都在，地基齐备，只是 sync trainer 没接通。

### 4.3 现状语义校准（关键，纠正此前表述）
- 用户说"现状跑分离" + "分离+on-policy 尚未跑通" → **"跑分离"是目标态/计划，非已落地现状**。
- 分离走的代码路径**尚未确定**（要求社区易接受）。
- 训练侧 Megatron 是否已跑通（作为已验证基座）→ 待确认。
- **项目真实起点 = 从零打通 "on-policy sync + 分离推理" 这条路**，再在其上加多任务借调。无分离则无可借资源。

## 5. 目标场景：公共资源池模型（用户逐字确认）

### 5.1 场景描述
1. 先启动一个 Ray 集群，**初始无任何 RL 任务，是公共资源池，不属于任何任务**。
2. 多个 RL 任务**不定时加入**；每个任务**自带资源（卡）**，把资源**注册进公共集群**，然后启动训练。
3. 任何一个 RL 任务可**将自己的资源贡献给其他任务**；公共池的资源会**分配给其他任务使用**。
4. 任务退出时其资源**需回收**（含崩溃/异常的可靠性）。

### 5.2 池化落地约束（5 子问，用户已答）
| 子问 | 用户答 | 含义 |
|------|--------|------|
| 注册粒度/层级 | 任务自带**卡(GPU)**；协调器在 **Ray 资源管理层之上、不替换 Ray**，只管"资源给谁用" | Ray 做底层资源面，协调器做上层调度决策面，分层 |
| 贡献策略 | 有空闲卡就贡献**直到为 0**（全捐，不留自留额度）；需时待协调器分配 | 激进全捐，池化效益最大化 |
| 借出形态 | 池借出**裸卡**；借入方**重拉推理实例或预热** | 冷启动由借入方承担，无热接 |
| 退出回收 | 需回收资源 + 可靠性（崩溃/异常释放与泄漏防护） | — |
| 调度策略 | 核心调度策略**现阶段暂不考虑** | 先打通机制，调度算法后置 |

### 5.3 协调器角色定义
- = **公共资源池管理者/调度器**：维护池、接受任务注册与资源贡献、按需把池资源分配给繁忙任务、空闲/退出时回收。
- 任务与资源**解耦**：资源不"属于"任务，而是"任务贡献给池、池再分配给任务"。"纳管空闲推理卡"= 空闲卡回收到公共池，而非停留在原任务名下。
- 分层：**Ray 管资源分配（底层），协调器管资源归属/调度决策（上层）**。

### 5.4 两层关系
```
第一层（物理前提）：分离推理 —— 让推理卡能脱离训练、独立闲置/可回收
        ↓ 只有独立闲置的卡才能"回收到池"
第二层（协调器）：公共资源池 —— 接受注册与贡献、统一纳管/分配/回收
```
分离推理（§4）是池化（§5）的物理前提。无分离则无池化。

## 6. 核心矛盾

### 6.1 主矛盾：目标组合 verl 不支持
"GRPO + on-policy + sync + 分离"组合 v0.8.0 现成走不通（§4.2），分离地基在但 sync trainer 没接通。需在"不改 verl / 社区易接受"约束下打通——是项目第一步必解前置工程。

### 6.2 次级矛盾：全捐 + 裸卡 + on-policy 三者叠加
- **冷启动门槛刚性**：借出裸卡 + on-policy → 借入方重拉实例须加载**当前 step 权重**，权重加载是冷启动刚性项，无法靠预热绕过。**借调收益 = 空泡时长 − 冷启动成本**，此不等式不成立则方案前提不成立（→ G6 关键）。
- **防饥饿 × 一致性双重压力**：全捐到 0 → 任务自己 generation 时可能拿不回足够卡；on-policy 又要求借回的卡重载正确权重。
- **权重加载刚性 vs 裸卡**：除非池里维护"该任务最新权重常驻实例"，但这与"借出裸卡"形态矛盾。

### 6.3 候选方案线索（调研后画像更新，2026-07-10）
| 线索 | 思路 | 碰 verl 源码 | 社区接受度 | on-policy 适配 | 外部可控性 | 状态 |
|------|------|-----------|----------|---------------|----------|------|
| L1 | sync trainer 不改源码传 worker_group=None 的 config 开关/子类 hook | 0（外部子类+FQN） | 中-高（verl 有此机制） | 要绕 `:712` 硬传+`:334` assert + 换 backend 非 naive | 差（PPOTrainer 非 actor，外部碰不到） | 调研完，可走"子类+FQN+backend 切换" |
| L2 | 外部独立 rollout server + 外部权重注入，trainer 不感知 rollout 物理位置 | 0 | 低（自建旁路，与 verl 分叉） | 协调器自保证一致性 | 好（协调器全控） | 调研完，碰不到 sync 任务内部 |
| L3 | 复用 verl experimental 分离模块 | 0（用 experimental） | 中（复用 verl 能力） | 现成全绑 off-policy/async，要从 off-policy 改造 | 好（fully_async 有 remote actor add/remove_replicas） | 调研完，有外部可控能力但绑 async |

**调研新增结论**（详见 `01-verl-training-topology.md`）：
- verl 无 callback/hook 体系，但有 5 个 **FQN 类替换**扩展点（task_runner_class / agent_loop_manager_class / checkpoint_manager_class / custom_backend_module / rollout.name），外部插件可"写子类+config 挂载"接入，**不必直接改 verl 源码**。这是 P1/P2 现实落脚点。
- sync+STANDALONE 两道硬卡：`main_ppo_sync.py:712` 硬传 worker_group + `ray_trainer.py:334` assert hybrid；且 naive backend 不支持跨进程（要切 nccl/nixl）。
- sync 路径 PPOTrainer 是 driver 进程内普通对象（非 Ray actor），外部无法 RPC → 协调器碰不到 sync 任务内部，须自建编排层。
- 最大推理空闲段（空泡主战场）：step 内 `sleep_replicas`→`update_weights` 之间（`main_ppo_sync.py:1712→1651`）。

## 7. 背景缺口（待补充）

> 以"能否直接喂给 PoC 设计"为标准。按影响程度排序。

**一类（改变架构走向）：**
- **G3 空泡量级与相位**：空泡占端到端耗时几成？绝对时长（秒/分钟级）？多任务天然错峰还是撞车？→ 决定搬卡 vs 错峰优先级、借调冷启动阈值。**与 §6.2 直接挂钩，最高优先级。**
- **G4 模型同构性**：多任务是否同一 base model/同尺寸？跨模型借调冷启动与一致性代价更高。
- **G6 借调冷启动成本基线**：vLLM rollout init→出 token 耗时？权重全量加载耗时？→ 决定 §6.2 不等式是否成立。

**二类（影响可行性边界）：**
- **G5 权重同步现状 + 调试文件故事**：工作区 `分析RL训练材重加载逻辑.md`/`dump_vllm_checksums.py`/`trae-debug-log-first-rollout-garbled-weights.ndjson` 疑排查 trainer→rollout 权重同步坑（首轮权重错乱）。是否项目直接前置痛点？
- Megatron 训练侧是否已跑通（作为基座）。

**三类（偏验收/交付，最后再说）：**
- **G7 交付范围**（设计/PoC/路线）：背景齐了自然有倾向。
- **G8 验收量化目标**：耗时降% / 利用率提% / 空泡降%。

## 8. 调研计划（代码侧，v0.8.0 基线）

> 目的：(1) 端到端梳理 verl 训练逻辑全貌（组件关系/部署形态/时序），不深挖细节；(2) 量化 L1/L2/L3 各自"碰多少 verl/社区接受度/借调代价"，喂给 P1 定界。

- [ ] **T0：端到端训练逻辑全貌** — 主控 loop（generation/weight-sync/training 边界与可观测点）、组件间关系、部署形态、step 时序。
- [ ] **T1：vLLM rollout 生命周期** — 实例启停/伸缩/运行时增减 GPU 可行性；冷启动成本构成。
- [ ] **T2：权重同步机制** — trainer→rollout push 路径；借调卡归还避免脏权重；增量/异步加载可能。
- [ ] **T3：Ray 资源管理抽象** — PG/actor 模型；外部不替换 Ray 能否做"资源给谁用"决策；任务退出回收机制。
- [ ] **T4：多任务/多租户** — 同集群多 driver 现状；多任务资源隔离与纳管面。
- [ ] **T5：L1/L2/L3 可行性量化** — 每条要碰多少 verl 源码/社区接受度/借调代价。

调研结论回填到 §9「verl 扩展点清单」与 §6.3 线索状态。

## 9. verl 扩展点清单（调研回填）

> verl **无 callback/hook 体系**，但暴露 5 个 **FQN 类替换**扩展点（config 填全限定类名 → 运行时反射加载外部子类）。这是"不改 verl 源码、插件式接入"的现实集成面。详见 `01-verl-training-topology.md` §E14。

| 扩展点 | 位置 | 用途 | 接入方式 | 需改 verl？ |
|-------|------|------|---------|----------|
| `task_runner_class` | `main_ppo.py:52/82`、`main_ppo_sync.py:1862` | 换 TaskRunner 子类（可覆写 init_workers 等） | FQN config | 否 |
| `agent_loop_manager_class` | `main_ppo_sync.py:716-720` | 换 rollout 管理器子类 | FQN config | 否 |
| `checkpoint_manager_class` | `ray_trainer.py:960-964` | 换 CheckpointEngineManager 子类（控权重同步/扩缩容） | FQN config | 否 |
| `checkpoint_engine.custom_backend_module` | `base.py:287-303`、`rollout.yaml:290-292` | 注册新权重传输 backend | import 外部模块+装饰器注册 | 否 |
| `rollout.name` → `RolloutReplicaRegistry` | `replica.py:302-310` | 换 replica 实现 | 注册 | 否 |

**缺口（需自建，verl 不提供）：**
- 实时 phase 探测：verl 只有事后 wandb timing，无实时 RPC 查"当前在 gen/training"。→ 协调器须靠子类覆写 step 自埋 hook。
- 外部 RPC 触发 sync 任务扩缩容/权重重同步：sync PPOTrainer 非 Ray actor，外部碰不到。→ 须自建编排层（L2）或换 async 路径（L3）。
- 资源回收兜底：verl 不 kill actor/不 remove PG，靠 Ray GC。→ 协调器须自建回收。
- 多任务编排：verl 天然单任务，无 task/job 抽象。→ 协调器须自建多 driver 编排层。

## 10. 验收标准（待对齐）

| # | 指标 | 目标值 | 状态 |
|---|------|--------|------|
| A1 | 端到端训练耗时下降 | ?% | [ ] |
| A2 | 推理卡利用率提升 | ?% | [ ] |
| A3 | 空泡占比下降 | ?% | [ ] |
| A4 | 不破坏 verl 单任务正确性 | 0 回归 | [ ] |

## 11. 决策记录
- 2026-07-10：确立 P1-P3 硬原则；进入背景约束对齐。
- 2026-07-10：基线切精确 v0.8.0 tag（7aed6b23）；撤回 release/v0.8.0 顶端基线的 3 个 agent，待 v0.8.0 基线重启。
- 2026-07-10：确认核心约束组合 = GRPO + on-policy + sync + 分离(STANDALONE)；代码核实 v0.8.0 现成不支持（§4.2），但分离地基在。= 项目核心矛盾（§6.1）。
- 2026-07-10：记录候选线索 L1/L2/L3（§6.3），仅记录不验证。
- 2026-07-10：G1 确认现状跑分离 → 续澄清：分离路径未定(要社区易接受)+分离+on-policy 尚未跑通 → "跑分离"=目标态非现状 → 项目起点=从零打通 on-policy sync+分离（§4.3）。
- 2026-07-10：P1 边界暂不定，先调研再定界（§2/§5.2）。
- 2026-07-10：G2 明确公共资源池模型（§5），协调器=公共池管理者，Ray 上层调度决策层；池化落地 5 子问已答（§5.2）。
- 2026-07-10：识别核心次级矛盾——全捐+裸卡+on-policy 三叠加，引出借调冷启动门槛（§6.2）。
- 2026-07-10：重组对齐文档为现行结构；流水原文归档至 `00-alignment.raw-archive.md`。启动 v0.8.0 基线端到端调研（T0 优先）。
- 2026-07-10：端到端调研完成（详见 `01-verl-training-topology.md`）。核心产出：verl 无 callback 但有 5 个 FQN 类替换扩展点（task_runner/agent_loop_manager/checkpoint_manager/custom_backend/rollout.name）→ 外部子类+config 挂载可接入，不改源码。L1/L2/L3 画像量化（§6.3）。两道硬卡确认（`:712`+`:334`+naive backend）。空泡主战场定位：step 内 `sleep_replicas`→`update_weights`（`:1712→1651`）。三处 verl 不兜底（实时 phase/外部 RPC/资源回收/多任务编排）需协调器自建。下一步：据 §6.3 画像定 P1 边界 + 选定主线线索。
