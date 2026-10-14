# 大模型能力评测方法论与五家国产厂商分层对比报告

**生成日期**: 2026-10-14  
**研究对象**: 通义千问 Qwen、深度求索 DeepSeek、字节豆包 Doubao、月之暗面 Kimi、智谱清言 GLM  
**证据基准**: 仅使用公开的第三方评测榜单与厂商公开技术材料；不调用任何模型 API，不进行实测复现，不访问任何厂商控制台  
**数据快照**: Artificial Analysis 2026-10-14、LiveBench 2026-06-25、Terminal-Bench 4.0（2026-08/09 条目）、SWE-bench 官方榜（2026-10-14 抓取，最新条目 2026-02）、LMArena（二手快照 2026-10-08 与周榜第 41 周）、SuperCLUE 2026-08

> **本报告的一条硬规则**：所有分数均标注来源与日期，并区分「第三方实测 / 官方榜单 / 厂商自报 / 二手聚合」。**凡未找到证据者一律标注「无证据」，不做任何估算、插值或"合理推测"。** 排名的置信度与"我们不知道什么"与排名本身同等重要。

---

## 0. 摘要（TL;DR）

1. **当前没有单一可信的"能力总分"。** 同一批模型在 Artificial Analysis 复合指数与 LiveBench 上的相对位次差异极大（例：DeepSeek V4.1 Flash 在 AA 指数上落后 Qwen3.8 Max 约 6 分，在 LiveBench 上却反超 2 分）。任何把多个榜单原始百分比直接平均的做法都会产生系统性错误结论。
2. **本报告提出的量规为 8 维加权**（推理 20 / 编码 20 / 长上下文 8 / 工具智能体 12 / 中文 12 / 指令遵循 8 / 延迟成本 10 / 稳定性 10），并要求**每维单一首选仪器、不做跨仪器混合平均**。
3. **五家厂商的第一梯队为 GLM-5.3、Qwen3.8-Max、Kimi K3**（各在其中至少一个维度领先），**第二梯队为 DeepSeek V4.1 Flash / GLM-5.3-Flash / Qwen3.8-Flash-Next 等快档**，DeepSeek V4 Pro 处于两梯队之间。
4. **豆包（Doubao）无法参与排名**：在西方第三方榜单上几乎完全缺席，公开数字绝大多数为字节自报，且独立聚合器数据与自报数据冲突到无法调和（SWE-bench Verified 自报体系 78.8 vs 聚合器 36）。
5. **最大的三个陷阱**：① 网关别名与实测模型的代际错位（多数别名落后 1–2 代）；② 榜单版本/口径漂移（AA 指数已迭代到 v4.3.2，权重从等权 → 25/25/25/25 → 30/20/20/30）；③ 排名对缺失值处理策略不稳健（见 §4.4）。

---

## 1. 评测方法论全景（2026 年 10 月）

### 1.1 人类偏好竞技场：LMArena / Chatbot Arena

| 项目 | 内容 |
|---|---|
| 测什么 | 匿名两两盲测下的人类**偏好**，不是"能力"。2023 年至今累积千万级投票 |
| 计分 | Bradley-Terry 极大似然估计（2024-01 起由在线 Elo 改为全量 BT MLE）；`arena-rank` 官方实现用协方差/自助法给出 95% 置信区间 |
| 风格控制 | 2024-08-28 起把**答案长度、markdown 标题/加粗/列表**作为 BT 协变量。控制后 GPT-4o-mini、Grok-2-mini 名次明显下滑，Claude 3.5 Sonnet/Opus 上升 |
| 有效性边界 | **长度/冗余偏好**（一项分析称 5 个百分点的"偏好更长"偏差 ≈ 35 Elo 等价差距）；**刷票与选择性采样**（"The Leaderboard Illusion" 记录头部两家厂商占据约 19.2%/20.4% 的对战数据，且存在私有测试与"best-of-N 检查点挑选"）；**同时置信下相邻名次多数不显著**（一项基于 106,134 票的研究发现，54 组相邻配对中仅 1 组在多重比较校正后仍然显著） |
| 建议权重 | **中**。是唯一大规模直接人类偏好信号，但**只应引用带风格控制的榜单**，且只在相同提示分布内比较 |

> 本报告实测限制：`lmarena.ai` 在本环境网络不可达，LMArena 数据**只能引用二手快照**（见 §3），并已显著降权。

### 1.2 抗污染在线基准

| 仪器 | 测什么 | 有效性边界 | 建议权重 |
|---|---|---|---|
| **LiveBench** | 每月刷新的客观题（arXiv/竞赛/新闻/Kaggle/IMDb 来源），**无 LLM 裁判**，约 1/6 题目每月替换 | 仍是"客观题"，不测开放式agent 能力；公开 CSV 只给逐任务分，**官方综合分是站点自己算的类别均值，与逐任务均值不同**（本报告用逐任务均值并已声明） | **强** |
| **LiveCodeBench** | 持续抓取 LeetCode/AtCoder/Codeforces，按**时间切分**只计模型知识截止之后发布的题目 | "免污染"是有条件的；`release_v6`（1,055 题，≤2025-04）长期未更新，切后样本小、差异不具统计意义 | **强（仅限算法编码）** |
| **ARC-AGI-2** | 网格抽象推理，pass@2，题目由人类验证可解，并从设计上"防猜" | 与本文五家厂商**无公开数据**（本报告核查 arcprize.org 排行榜后确认无条目）→ 只能作为"未知项" | 强，但本报告用不上 |

### 1.3 推理 / 数学

| 基准 | 2026 年状态 | 边界 |
|---|---|---|
| AIME 2025/2026 | **已饱和**：头部模型报告 100%（AIME 2025）与 99%+（2026） | 失去区分度；AIME 2024 已被发现有污染证据 |
| MATH-500 | 饱和（99%+） | 审计发现 7 处训练/评测重叠，但移除后名次变动 <0.3 分 |
| GPQA Diamond | 接近饱和：前沿区间约 91–95%（博士专家约 65–70%）。Epoch AI 已于 2026-05 宣布其饱和 | **不建议作为主判据**；本报告仅作佐证 |
| FrontierMath（Epoch AI） | 仍未饱和（原始题从 <2% 提升到约 48–52%）；私有、可自动验证、防猜 | 公开材料中**未找到本文五家厂商的分项成绩**；且 Epoch 注明 o3 的"25%"在标准算力下无法复现 |

**关键证据——"新题悬崖"**：在结构匹配但发布在知识截止之后的题目上，Grok 4 Heavy 从 76.2% 掉到 11.1%、o3 从 52.4% 掉到 15.1%。这说明"静态基准高分"与"真实能力"之间存在系统性缺口。

### 1.4 编码智能体

| 基准 | 说明 | 边界 |
|---|---|---|
| **SWE-bench Verified** | 500 个工程师核验过的 PR 任务 | **OpenAI 于 2026-02-23 宣布停止报告该指标**：审计 138 个稳定失败任务，59.4% 存在实质问题（35.5% 测试过严、18.8% 超出范围），并确认存在污染（部分模型可从记忆复现金标补丁）。Epoch AI 于 2026-09-03 评为 "Flawed" |
| **SWE-bench Pro**（Scale AI） | 1,865 题，公开/私有/留出三分，指标为 Resolve Rate（须通过 fail-to-pass 且不破坏 pass-to-pass） | 公开→私有分数下滑是污染信号；2026 年审计发现约 30% 任务损坏及 `.git` 历史利用漏洞 |
| **Terminal-Bench 2.x / 4.0** | 真实容器内 CLI 任务，**只看最终容器状态判分**，三人核验（特异性/可解性/完整性），每个任务约 3 人时 | 受脚手架影响大（官方同时报告 best-fit 与中性 Terminus 脚手架）。**本报告将其作为"编码/智能体"维度的首选仪器** |

### 1.5 长上下文

| 仪器 | 边界 |
|---|---|
| NIAH（大海捞针） | **已饱和**（1M 上下文 >90–95%），且不能预测多跳推理 |
| RULER（NVIDIA） | 13 项任务；前沿模型实际可稳定使用的是标称窗口的约 50–65% |
| MRCR / MRCR v2 | 多针版本分数塌陷明显（如某前沿模型 1M 处 8 针仅 26.3%） |
| NoLiMa | 去除字面关键词重叠后，11 个 SOTA 模型在 32K 处跌破短上下文的 50% |
| AA-LCR（Artificial Analysis） | 第三方实测的长上下文推理子分，**本报告采用**（唯一对本文五家厂商都有覆盖的长上下文第三方实测） |

### 1.6 工具 / 智能体

- **τ²-Bench**：多轮、受策略约束、用户由 LLM 模拟；同时报告 pass@1 与 **pass^k**（全 k 次通过），后者衡量可靠性——这是它最有价值的设计。
- **BFCL v3/v4**（Berkeley）：v3 增加多轮/状态化，v4 大幅加权 agentic（联网检索、记忆、格式敏感）。主要测**单次调用形态正确性**，不测跨试验可靠性。注意：BFCL 大量出现在厂商模型卡里，存在训练集污染担忧。
- **GDPval-AA / AA-Briefcase / AutomationBench-AA**（Artificial Analysis 的 agent 类评测）：以真实经济任务为主，GDPval-AA 用 Elo 且锚定冻结。**本报告"工具智能体"维度首选 AA 的 GDPval 子分**。

### 1.7 中文与多语言

- **C-Eval**（13,948 题 / 52 学科）、**CMMLU**（11,528 题 / 67 学科）、**CLUE**（2020 年，类 GLUE）：静态公开题库，**污染与"选择题 ≠ 能力"问题明确**；且**本文五家厂商当前旗舰在这些榜上均未找到新成绩**（仅能找到 Qwen2.5 时代数据）。
- **SuperCLUE**：中文通用榜，由数学推理、幻觉控制、科学推理、精确指令遵循、智能体编程、智能体任务规划六项构成，使用裁判模型打分。一项榜单一致性研究给出其平均一致性 63.6%（在其比较的 10 个榜单中最高）。**本报告"中文"维度采用 SuperCLUE**，但明确标注为**二手来源**。
- **Chinese SimpleQA**：中文事实性基准，按 CO/NA/IN/CGA 与 F 值评分。ACL 2025 论文中：Doubao-pro-32k 61.9/65.3，GLM-4-Plus 58.7/60.9。**仅覆盖上一代模型**。

### 1.8 指令遵循

- **IFEval**：25 条可验证格式约束；**已饱和**（小模型也 >80%），因为训练数据常直接按 IFEval 分类法合成。
- **IFBench**（300 提示 / 58 条新约束 / 每条带 Python 验证器，用留出的 WildChat 提示防污染）：**前沿模型普遍 <50%**，是对未见约束的泛化测试。**2026 年应优先使用 IFBench 而非 IFEval。** 但本文五家厂商的当前旗舰在 AA 的 IFBench 子分上**大量为无数据**（详见 §3.4）。

### 1.9 幻觉 / 事实性

- **SimpleQA / SimpleQA Verified**：闭卷短事实问答；不同追踪站口径不一，分数差异极大。
- **HaluEval**：无实时榜单，属研究标准。
- **Vectara HHEM**：语料已扩到 7,700+ 篇、最长 32K；**必须核对语料版本**（不同版本分数不可比）。
- **AA-Omniscience**：6,000 道开放题，同时给出准确率与**幻觉率**，是"该说不知道时是否承认"的少数称职测量。注意其"准确率越高、幻觉率也可能越高"的激励错位。

### 1.10 聚合器

| 聚合器 | 方法 | 可复现性 |
|---|---|---|
| **Artificial Analysis Intelligence Index** | **v4.3.2（当前版本）**：10 项评测 → 4 大类**加权**：Agents **30%**、Coding **20%**、Scientific Reasoning **20%**、General **30%**。逐项权重：AA-Briefcase v1.1 15%、GDPval-AA v2.1 10%、AutomationBench-AA 5%（Agents）；Terminal-Bench 4.0 10%、SciCode 10%（Coding）；AA-Omniscience 15%、GDP.pdf 10%、AA-LCR v1.1 5%（General）；HLE 10%、CritPt 10%（Sci. Reasoning）。**纯文本、英文**。 | 自建自测，但**保留私有测试变体与过滤规则**，非完全可复现 |
| **Stanford HELM** | 分解为 Aspect/Scenario/Adaptation/Metric；用 **mean win rate** 而非均值分，原因是各任务度量尺度与方向不同 | 提示级全透明，可复现性最好 |
| **Epoch AI** | 用开源 Inspect 框架重跑；预处理把随机猜测映射到 0、已知错误率上限映射到 1；**ECI 复合分**用品格反应理论（IRT）拟合 61 个基准 / 274 个模型 | 公开日志与方法，独立性最好 |
| **OpenRouter** | 按 token 用量排名（**不是能力测量**） | 公开，可作需求量参照 |

### 1.11 如何把它们合成一个"站得住脚"的排名

**这是本报告最核心的方法论主张。**

**(A) 为什么原始百分比求平均会误导**

1. **难度尺度不同**：把饱和的 AIME（100%）与基本无解的 ARC-AGI-2（个位数）平均，等于让简单榜主宰结论。
2. **覆盖率偏差会奖励"逃避难题"**：有实证案例——只做 3 个简单榜的模型得 83.5 分，**高于**做了 18 个榜（含最难）的 66.7 分模型；加最小数量门槛也修不好，因为问题在于"**做了哪些榜**，而不是做了几个"。
3. **相关评测重复计数**：GPQA 等同时出现在多个复合分中。
4. **算术平均本身是隐藏的选择**：有研究称之为"沉默的默认值 ρ≈0.5"，会造成**置信膨胀**（聚合后的置信度高于最弱单一信号）；在 54 个前沿模型 × 10 个 HELM 场景上，"按均值的前 5"与"按最弱环节的前 5"**完全不重叠**。
5. **缺失数据的处理策略会直接决定名次**（本报告 §4.4 用真实数据演示了这一点）。
6. **时效性**：静态榜在数月内被污染，分数是"会过期的知识主张"。

**(B) 归一化选择**（据 JRC/OECD 复合指标手册）

- **min-max**：简单、透明，但对离群值敏感、不居中。
- **z-score**：保留分布形状，但对离群值敏感，且产生负值会破坏几何聚合。
- **百分位/秩**：统一量纲但**丢弃水平差异**；该手册明确指出**对百分位做算术平均是不可接受的**。
- 结论：归一化/加权/聚合的选择**不是中性的**，因此复合分**不得被当作客观量呈现**，必须附敏感性检验。

**(C) 本报告采用的合成规则**

1. **每个维度只允许一个首选仪器**（避免把 Terminal-Bench 4.0 与 2.1 混入同一列——本报告在初稿中正是踩了这个坑，详见 §4.4）。
2. **允许在同一维度并列展示次要仪器作为佐证，但不参与打分**。
3. **缺失值不重归一化权重、不填 0**；改用**同维度已观测模型的中位数做中性填补**，并**同时给出"证据覆盖率"**。
4. **同时报告两种聚合算子**：加权均值（均值口径）与最弱环节（min 口径，保守口径）。
5. **必做敏感性分析**：若更换缺失值策略会改变名次，则该名次不得被当作结论。

**(D) 可辩护的记分卡长什么样**

> 一张"每维得分 + 每维置信度 + 证据覆盖率 + 数据来源与日期"的矩阵，**而不是一个总分**。总分只能作为辅助，且必须附上"在何种假设下成立"。

---

## 2. 本报告提出的评分量规（Rubric）

### 2.1 维度与权重

权重按"本地多协议网关的选型/路由决策"取向设定，而非按学术界的通用能力取向。

| # | 维度 | 权重 | 首选仪器（单一口径） | 次要佐证 |
|---|---|---|---|---|
| D1 | 推理与疑难问题求解 | **20%** | AA-HLE（第三方实测，%） | GPQA-D、AA-CritPt、FrontierMath |
| D2 | 编码与软件工程 | **20%** | Terminal-Bench 4.0 官方榜（%） | SWE-bench Multilingual（checked）、AA Terminal-Bench 2.1 |
| D3 | 长上下文 | **8%** | AA-LCR v1.1（%） | MRCR、RULER、Fiction.LiveBench |
| D4 | 工具调用 / 智能体 | **12%** | AA GDPval-AA 子分（%） | τ²-Bench、BFCL v4、AA-Apex |
| D5 | 中文语言与文化 | **12%** | SuperCLUE 通用榜总分 | Chinese SimpleQA、C-Eval/CMMLU（本轮无新数据） |
| D6 | 指令遵循与结构化输出 | **8%** | AA IFBench（%） | IFEval（已饱和，仅回归看门） |
| D7 | 延迟与成本 | **10%** | AA 实测 tok/s + 输出价（第三方实测） | TTFT、缓存命中价 |
| D8 | 稳定性与可用性（含风控抗性） | **10%** | **当前无第三方可用性测量 → 本报告不评分** | OpenRouter uptime、本网关风控台账 |

**合计 100%。** D6 因当前旗舰普遍缺 IFBench 数据、D8 完全无第三方测量，在实际计算中权重被**声明为"暂缺"**而非重分配（重分配会引入覆盖率偏差，见 §1.11 A-2）。

### 2.2 0–100 分标定与锚点

由于首选仪器本身就是 0–100 的百分比（正确率/通过率），**本报告直接采用原始百分比作为维度分，不做二次缩放**。锚点定义如下：

| 分数区间 | 含义 | 判据 |
|---|---|---|
| 90–100 | 全球前沿级 | 在有区分度的仪器上进入全球前 5% |
| 75–89 | 第一梯队追赶者 | 与全球前沿差 1 个基准档，国产前 3 水平 |
| 60–74 | 强商业级 | 生产可用，最难任务上差距清晰 |
| 45–59 | 可用但有明确短板 | 需按任务路由 |
| 30–44 | 受限（快档/小参数） | 仅在低成本轻任务上合适 |
| 0–29 | 该维度能力不足或无证据 | — |
| **无证据** | **不作分** | 不填 0、不插值 |

### 2.3 置信度标定

| 置信度 | 条件 |
|---|---|
| **高** | ≥2 个独立第三方仪器同时覆盖，且被测版本号可与网关别名对应 |
| **中** | 1 个第三方仪器，或 1 个官方榜单 + 少量交叉验证 |
| **低** | 仅有厂商自报，或仅有二手聚合器，或版本/别名无法对应 |

**关键**：**置信度不是分数的修饰词，而是分数的组成部分**。一个"85 分 / 低置信"与"78 分 / 高置信"在路由决策中的含义完全不同。

### 2.4 延迟/成本子分的透明算法

为避免"感觉性打分"，D7 用可复现公式：

- `速度分 = 100 × min(tok/s ÷ 200, 1)`（200 tok/s 记为满分）
- `价格分 = 100 × clamp( log10(15 ÷ 输出价) ÷ log10(15 ÷ 0.2), 0, 1 )`（输出价 $0.2/M 记满分，$15/M 记 0）
- `D7 = (速度分 + 价格分) ÷ 2`

本次计算结果：DeepSeek V4.1 Flash 76.1；GLM-5.3-Flash 56.1；Qwen3.8-Flash-Next 52.9；DeepSeek V4 Pro 0813 40.3；GLM-5.3(Max) 29.7；Qwen3.8-Max(0902) 21.2；Kimi K3(Max) 11.2。

---

## 3. 证据表

> **图例**：[3P] 第三方实测 · [OFF] 官方榜单 · [V] 厂商自报 · [2ND] 二手来源/聚合器 · **[无证据]** 明确未找到

### 3.1 映射关系与版本错位（先读这一节）

**网关别名与本轮实测模型不是同一代**——这是解读所有后续数据的前提。

| 网关别名 | 公开可实测的同代/后继模型 | 代际错位 |
|---|---|---|
| `qwen3.7-max` | Qwen3.8-Max / Qwen3.8-Max-0902 | 落后 1 代 |
| `qwen3.6-flash` | Qwen3.8-Flash-Next | 落后 2 代 |
| `qwen`（默认别名） | **没有任何公开评测对应"Qwen"这个别名** | 不可对应 |
| `deepseek-chat` / `deepseek-reasoner` | V3.2 档位；而公开实测的是 V4-Pro / V4-Flash / V4.1-Flash | **错位最大**：别名很可能仍是 V3.2 |
| `doubao-pro` / `-think` / `-lite` | Doubao-Seed-1.5/1.6/1.8/2.0/2.1 | 落后 1–2 代，且几乎无第三方数据 |
| `kimi` / `kimi-explore` / `kimi-math` | Kimi K2 / K2.5 / K2.6 / K3 | 落后 1–2 代；`kimi-explore`/`kimi-math` **无任何公开评测** |
| `glm-4-plus` / `glm-4-flash` / `glm-zero-preview` | GLM-4.5/4.6/4.7/5/5.1/5.2/5.3 | 落后 2–3 代；`glm-zero-preview` **无任何公开评测** |

### 3.2 Artificial Analysis Intelligence Index v4.3.2（[3P]，2026-10-14 抓取自 artificialanalysis.ai）

参照系：**69 个未弃用且指数 ≥30 的模型**；该参照系最高分为 **Claude Opus 5.5 (Max) = 57.6**。

| 模型 | AA 指数 | 是否估算 | 开源 | 上下文 | 输入/输出价 ($/M) | 输出速度 (tok/s) | TTFT (s) | 相对位次* |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.8 Max (0902)** | **45.4** | 否 | 否 | 983,616 | 2 / 6 | 42.3 | 2.96 | 58.0 |
| **GLM-5.3 (Max)** | **44.8** | 否 | 是 | 1,000,000 | 1.4 / 4.4 | 61.9 | 3.50 | 55.1 |
| **Kimi K3 (Max)** | **43.6** | 否 | 是 | 1,048,576 | 3 / 15 | 44.8 | 3.17 | 52.2 |
| GLM-5.3-Flash | 41.8 | 否 | 是 | 1,000,000 | 0.15 / 0.5 | 66.9 | 2.80 | 43.5 |
| Qwen3.8-Flash-Next | 39.8 | 否 | 是 | 256,000 | 0.15 / 0.47 | 51.2 | 2.01 | 34.8 |
| DeepSeek V4.1 Flash (Max) | 39.5 | 否 | 是 | 1,000,000 | 0.3 / 1.2 | 187.3 | 0.96 | 31.9 |
| **DeepSeek V4 Pro 0813 (Max)** | **36.0** | 否 | 是 | 1,000,000 | 1.32 / 3.96 | 99.4 | 1.62 | — |
| DeepSeek V4 Flash 0731 (Max) | 34.3 | 否 | 是 | 1,000,000 | 0.44 / 1.32 | 182.8 | 1.23 | — |
| GLM-5.2 (Max)（已弃用） | 33.7 | 否 | 是 | 1,000,000 | 1.4 / 4.4 | 82.7 | 2.97 | — |
| **Qwen3.7 Max**（已弃用） | **29.5** | 否 | 否 | 1,000,000 | 2.5 / 7.5 | 196.5 | 2.26 | — |
| Kimi K2.6 (Reasoning)（已弃用） | 27.0 | 否 | 是 | 256,000 | 0.95 / 4 | 47.0 | 2.78 | — |
| Kimi K2.7 Code | 25.8 | 否 | 是 | 256,000 | 0.95 / 4 | 63.9 | 2.75 | — |
| DeepSeek V3.2 (Reasoning)（已弃用） | 21.5 | **是** | 是 | 128,000 | 0.28 / 0.42 | 无 | 无 | — |
| **Doubao Seed Code** | **16.9** | **是** | 否 | 256,000 | 无 | 无 | 无 | — |

\* 相对位次 = 在 69 模型参照系中的百分位（仅作横向刻度，**不可**用于求平均，见 §1.11 B）。

**豆包的唯一 AA 条目是"估算值"**——即 Artificial Analysis 自己都没有对豆包做过完整实测。

### 3.3 AA 关键子分（[3P]，同一抓取）

| 模型 | GPQA-D | HLE | Terminal-Bench 2.1 | AA-LCR | SciCode | MMMU-Pro | GDPval-AA | CritPt | AA-Apex | τ²-Bench | IFBench |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8 Max (0902) | 92.8 | **43.1** | **88.8** | 80.3 | 52.1 | **82.8** | **58.3** | 17.7 | **42.4** | 无 | 无 |
| GLM-5.3 (Max) | 91.7 | 42.3 | 83.9 | 79.7 | **59.0** | 无 | 57.5 | 19.1 | 无 | 无 | 无 |
| Kimi K3 (Max) | **93.5** | **46.9** | 85.0 | **88.7** | **59.5** | 80.5 | 51.7 | **23.4** | 41.3 | 无 | 无 |
| GLM-5.3-Flash | 91.2 | 39.9 | 84.3 | 80.0 | 51.6 | 无 | 57.2 | 15.4 | 无 | 无 | 无 |
| Qwen3.8-Flash-Next | 92.3 | 38.0 | 86.1 | 79.7 | 50.6 | 79.8 | 56.5 | 11.1 | 无 | 无 | 无 |
| DeepSeek V4.1 Flash (Max) | 无 | 39.2 | 无 | 84.0 | 51.9 | 77.0 | 55.0 | 14.3 | 无 | 无 | 无 |
| DeepSeek V4 Pro 0813 | 92.8 | 41.0 | 78.7 | 80.3 | 51.0 | 无 | 47.5 | 18.0 | 无 | 无 | 无 |
| Qwen3.7 Max（弃用） | 92.3 | 40.5 | 74.5 | 79.0 | 49.5 | 无 | 31.4 | 13.4 | 无 | 94.7 | **80.5** |
| GLM-5.2 Max（弃用） | 89.5 | 41.1 | 77.9 | 78.3 | 51.2 | 无 | 43.6 | 20.9 | 33.7 | **99.1** | 73.3 |
| Kimi K2.6（弃用） | 91.1 | 37.5 | 65.9 | 81.0 | 51.5 | 79.4 | 26.9 | 8.0 | 28.5 | 95.9 | 76.0 |
| Doubao Seed Code | 76.4 | **14.1** | 无 | 71.7 | 无 | 68.1 | 无 | 0.3 | 无 | 58.2 | 51.4 |

**要点**：① 当前旗舰的 IFBench 与 τ²-Bench **普遍无数据**——即"指令遵循"和"工具调用"两个维度在本代旗舰上出现了测量真空；② 豆包子分与其自报数字（见 §3.7）差距悬殊。

### 3.4 LiveBench 2026-06-25（[OFF] 官方 CSV + 本报告自算逐任务均值）

官方发布的最新快照为 **2026-06-25**（下一版未发布）。官方 CSV 只给 23 个任务的逐任务分，**不含综合分**；下表的"均值"是本报告自行计算的 23 任务算术平均（70 个模型同口径），**非 LiveBench 官方综合分**。

| 模型 | 23 任务均值 | 70 模型内排名 |
|---|---|---|
| claude-fable-5-1-max-effort（参照上限） | 83.8 | 1 |
| **deepseek-v4.1-flash-max** | **81.4** | **11** |
| **kimi-k3** | **79.5** | **17** |
| **qwen3.8-max** | **79.5** | **18** |
| deepseek-v4-pro-0813 | 78.2 | 24 |
| deepseek-v4-flash-vision-exp | 77.7 | 29 |
| qwen3.8-flash-next | 77.3 | 31 |
| **glm-5.3** | **76.6** | **36** |
| qwen3.8-27b | 75.8 | 38 |
| deepseek-v4-flash-0731 | 74.8 | 43 |
| **qwen3.7-max** | **74.1** | **45** |
| glm-5.2 | 73.4 | 49 |
| deepseek-v4-pro（旧版） | 72.6 | 54 |
| glm-5.3-flash | 71.1 | 56 |
| kimi-k2.6-thinking | 70.9 | 57 |
| qwen3.6-plus | 69.0 | 60 |
| kimi-k2.7-code | 68.8 | 61 |
| deepseek-v4-flash（旧版） | 66.1 | 67 |
| qwen3.6-27b | 64.2 | 68 |
| **Doubao 全系** | **[无证据]** | — |

### 3.5 Terminal-Bench 4.0 官方榜（[OFF]，tbench.ai，23 条公开条目）

| 厂商 | 智能体脚手架 | 模型 | Accuracy | pass@2 | 日期 |
|---|---|---|---|---|---|
| Anthropic | Claude Code | Opus 5.5（榜首参照） | 64.85 | 0.7318 | 2026-09-22 |
| **Z.ai** | Claude Code | **GLM-5.3** | **41.82** | 0.5076 | 2026-08-14 |
| **Z.ai** | Claude Code | **GLM-5.3-Flash** | **35.76** | 0.4652 | 2026-08-26 |
| **Alibaba** | Claude Code | **Qwen3.8-Max-0902** | **26.97** | 0.3924 | 2026-09-02 |
| **DeepSeek / Moonshot / ByteDance** | — | — | **[无证据]** | — | — |

**这是本报告中最"干净"的编码智能体信号**：结果判分制、有官方脚手架披露、且给出了 n_trials=330。GLM 在此项上明显领先 Qwen，而 DeepSeek、Kimi、豆包**完全缺席**。

### 3.6 SWE-bench 官方榜（[OFF]，swebench.com，2026-10-14 抓取；最新条目为 2026-02）

| 分榜 | 厂商 | 模型 | Resolved | 智能体 | 是否官方核验 | 日期 |
|---|---|---|---|---|---|---|
| Verified | ByteDance | Doubao-Seed-Code | **78.8** | TRAE | **未核验** | 2025-09-28 |
| Verified | Moonshot | Kimi K2 | 71.2 | Lingxi v1.5 | 未核验 | 2025-10-14 |
| Verified | Alibaba | Qwen3-Coder-480B-A35B | 69.6 | OpenHands | 未核验 | 2025-08-05 |
| Verified | Z.ai | GLM-4.6 | 68.2 | Undisclosed | 未核验 | 2025-09-30 |
| Multilingual | Z.ai | GLM 5 | **69.7** | mini-SWE-agent | **已核验** | 2026-02-13 |
| Multilingual | Moonshot | Kimi K2.5 | 67.3 | mini-SWE-agent | **已核验** | 2026-02-13 |
| Multilingual | DeepSeek | DeepSeek V3.2 | 59.0 | mini-SWE-agent | **已核验** | 2026-02-13 |
| bash-only | Moonshot | Kimi K2 Thinking | 63.4 | mini-SWE-agent | **已核验** | 2025-12-10 |
| bash-only | DeepSeek | DeepSeek V3.2 Reasoner | 60.0 | mini-SWE-agent | **已核验** | 2025-12-01 |
| bash-only | Z.ai | GLM 4.6 | 55.4 | mini-SWE-agent | **已核验** | 2025-12-01 |

**重要观察**：SWE-bench Verified 榜上**没有** Qwen3.8 / Kimi K3 / GLM-5.3 / DeepSeek V4 的任何条目——最新条目停留在 2026-02。这与 §1.4 中"OpenAI 停止报告该指标、Epoch 评为 Flawed"一致。**因此本报告不采用 SWE-bench Verified 作为评分仪器**，仅作历史沿革佐证。

### 3.7 LMArena Elo（[2ND]，二手快照，因 lmarena.ai 网络不可达）

**注意：以下为二手来源，置信度低。**

| 模型 | 总分榜 Elo（2026-10-08） | 名次 | 总分榜 Elo（周榜第 41 周，10-05~10-11） | Code 榜 | WebDev 榜 |
|---|---|---|---|---|---|
| qwen3.8-max | **1481.78** | 16 | 1483 | #29 / 1524 | **#10 / 1674**（0902 版） |
| kimi-k3-max | 1475.92 | 23 | **1488** | **#10 / 1540** | 无 |
| glm-5.3-max | 1471.40 | 26 | 1478 | 无 | 无 |
| glm-5.3-flash | 1469.62 | 30 | 无 | #27 / 1525 | 无 |
| deepseek-v4.1-flash-max | 1462.45 | 37 | 无 | #26 / 1528 | 无 |
| deepseek-v4-pro | 1451.00 | 48 | 无 | 无 | 无 |
| （参照上限）gemini-4-argon-high | 1533.46 | 1 | — | — | — |

两个快照对同一模型给出的名次相差最多 7 位（kimi-k3-max：#23 vs #16），**这本身就是 Elo 置信区间重叠的直接证据**，不应据此做精细排序。

### 3.8 中文榜：SuperCLUE 2026-08（[2ND]）

| 模型 | SuperCLUE 总分 | 备注 |
|---|---|---|
| Qwen3.8-Max（2026-08-03 版） | **71.48** | 0902 版为 **72.62**（当期权第一） |
| GLM-5.3（2026-08-14） | 71.29 | 智能体编程与任务规划两项领先 |
| Kimi K3（2026-07-16） | 70.68 | 编程 75.79，国产最高 |
| DeepSeek-V4-Pro（官方版） | 70.10 | 预览版标注为 64.40 |
| Doubao-Seed-2.1-Pro | 65.94 | 非开源 |
| DeepSeek-V4-Flash | 65.60 | — |
| GLM-5.2 | 63.27 | — |

**中文 SimpleQA（[3P]，ACL 2025 论文，仅覆盖上一代）**：Doubao-pro-32k 61.9 / 65.3；GLM-4-Plus 58.7 / 60.9。

### 3.9 豆包（Doubao）专项：自报 vs 独立的巨大裂口

| 指标 | 字节自报 [V]（Seed 2.1 Pro，2026-06-23 发布） | 独立/聚合 [2ND] | 差异 |
|---|---|---|---|
| Terminal-Bench 2.1 | 71.0 | 无 | — |
| SWE-Pro | 57.5 | 无 | — |
| SciCode | 59.8 | 无 | — |
| GDPVal | 87.9 | 无 | — |
| MMMU-Pro | 81.6 | 无 | — |
| OSWorld | 78.8 | 无 | — |
| SWE-bench Verified | （体系内口径约 78.8，且未核验） | **36%**（RankLLMs 聚合器） | **不可调和** |
| GPQA Diamond | 无 | **47.9%**（RankLLMs） | — |
| AA Intelligence Index | — | 仅 "Doubao Seed Code" 16.9，且标注为**估算** | — |
| LMArena / LiveBench / RULER / MRCR / BFCL / IFEval | 无 | **全部无证据** | — |

**结论：豆包在方法论上不可评分。** 一个自报 78.8 与一个聚合器 36 相差一倍以上，且缺少任何可交叉验证的第三方仪器。

### 3.10 需求侧参照：OpenRouter 用量份额（[OFF]，2026-10-13 数据）

DeepSeek 24.5%（+14%）、OpenAI 18.6%、Google 17.8%、**Z.ai 13.4%（+107%）**、Qwen 5.4%、Anthropic 3.9%。

这是**需求量**而非能力指标，但"Z.ai 翻倍、DeepSeek 居首"反映了市场对"高性价比"的真实投票，可作 D7 维度的行为学佐证。

---

## 4. 排名

### 4.1 分层排名（主结论）

**综合分排序在本证据水平下不稳健**（证明见 §4.4），因此**主结论以分层给出**，层内不强行排序。

| 梯队 | 厂商 / 模型 | 领先证据 | 代表短板 | 置信度 |
|---|---|---|---|---|
| **T1 第一梯队**<br>（与全球前沿差距最小） | **智谱 GLM-5.3 (Max)** | AA 指数 44.8；**Terminal-Bench 4.0 41.82（五家最高）**；SciCode 59.0（并列最高）；SuperCLUE 71.29；开源权重；$1.4/$4.4 | LCR 79.7（本组最低）；HLE 42.3 | **高** |
| | **阿里 Qwen3.8-Max (0902)** | **AA 指数 45.4（五家最高）**；**SuperCLUE 72.62（最高）**；GDPval-AA 58.3（最高）；Apex 42.4（最高）；LMArena WebDev **#10 / 1674** | **Terminal-Bench 4.0 仅 26.97（明显落后 GLM）**；D7 21.2（$6/M 输出、42 tok/s） | **中**（TB4.0 与 AA 指数给出方向相反的信号） |
| | **月之暗面 Kimi K3 (Max)** | **HLE 46.9（最高）**；**AA-LCR 88.7（最高，领先第二名 4 分）**；**CritPt 23.4（最高）**；GPQA-D 93.5（最高）；LMArena Code **#10 / 1540**；开源 1M 上下文 | **D7 仅 11.2（$15/M 输出、44.8 tok/s，本组最差）**；无 TB4.0 条目 | **高**（推理/长上下文）/ **低**（编码无官方榜证据） |
| **T1.5 旗舰但略逊** | **DeepSeek V4 Pro 0813 (Max)** | AA 指数 36.0；LiveBench 78.2；HLE 41.0；LCR 80.3；GPQA-D 92.8；开源 1M；SWE-bench Multilingual 有已核验的 V3.2 系 59.0 | GDPval-AA 47.5（本组最低）；AA 指数落后 T1 约 9 分 | **中** |
| **T2 高性价比快档** | **DeepSeek V4.1 Flash (Max)** | **LiveBench 81.4（五家最高、全场第 11）**；**187.3 tok/s（最快）**；TTFT 0.96s（最低）；$0.3/$1.2；LCR 84.0 | AA 指数 39.5；**TB4.0 无条目** | **高**（速度/成本）/ **低**（编码） |
| | **智谱 GLM-5.3-Flash** | AA 指数 41.8；**TB4.0 35.76（五家第二）**；GDPval 57.2；$0.15/$0.5 | SuperCLUE 无条目 | **中** |
| | **阿里 Qwen3.8-Flash-Next** | AA 指数 39.8；TB2.1 86.1；LiveBench 77.3；$0.15/$0.47 | TB4.0 无条目；SuperCLUE 无条目 | **中** |
| **不可排名** | **字节 豆包 Doubao** | 仅厂商自报（Seed 2.1 Pro） | **西方第三方榜单近乎完全缺席**；自报与聚合器数据冲突 | **低** |

### 4.2 逐维得分矩阵（0–100，直接取首选仪器原始百分比）

| 模型 | D1 推理<br>(HLE) | D2 编码<br>(TB4.0) | D3 长上下文<br>(AA-LCR) | D4 智能体<br>(GDPval) | D5 中文<br>(SuperCLUE) | D6 指令<br>(IFBench) | D7 延迟成本 | D8 稳定性 | **覆盖率** |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Max (0902) | 43.1 | **27.0** | 80.3 | **58.3** | **72.6** | 无证据 | 21.2 | 无证据 | 82% |
| GLM-5.3 (Max) | 42.3 | **41.8** | 79.7 | 57.5 | 71.3 | 无证据 | 29.7 | 无证据 | 82% |
| Kimi K3 (Max) | **46.9** | 无证据 | **88.7** | 51.7 | 70.7 | 无证据 | 11.2 | 无证据 | 82% |
| DeepSeek V4 Pro 0813 | 41.0 | 无证据 | 80.3 | 47.5 | 70.1 | 无证据 | 40.3 | 无证据 | 82% |
| DeepSeek V4.1 Flash | 39.2 | 无证据 | 84.0 | 55.0 | 71.8 | 无证据 | **76.1** | 无证据 | 62% |
| Qwen3.8-Flash-Next | 38.0 | 无证据 | 79.7 | 56.5 | 无证据 | 无证据 | 52.9 | 无证据 | 70% |
| GLM-5.3-Flash | 39.9 | 35.8 | 80.0 | 57.2 | 无证据 | 无证据 | 56.1 | 无证据 | 70% |

（"无证据"格不计分、不填 0；D6/D8 全列为空，故实际参与计算的权重上限为 82%。）

### 4.3 复合分（辅助，非结论）

采用 §1.11 C 的中性中位数填补（推理 41.0 / 编码 35.8 / 长上下文 80.3 / 智能体 56.5 / 中文 71.3）：

| 模型 | 复合分 | 证据覆盖率 |
|---|---|---|
| DeepSeek V4.1 Flash (Max) | 54.3 | 62% |
| GLM-5.3-Flash | 51.9 | 70% |
| Qwen3.8-Flash-Next | 50.9 | 70% |
| GLM-5.3 (Max) | 50.8 | 82% |
| DeepSeek V4 Pro 0813 | 48.7 | 82% |
| Kimi K3 (Max) | 48.1 | 82% |
| Qwen3.8-Max (0902) | 46.7 | 82% |

**这张表如果被当作结论，就是错的。** 它让 DeepSeek V4.1 Flash 登顶、让 Qwen3.8-Max 垫底，而两者的差距主要来自"谁有编码维度的观测值"以及"延迟/成本权重"——不是能力差距。理由见下节。

### 4.4 敏感性分析：为什么不能采信单一排序

同一批数据，只更换缺失值处理策略：

| 策略 | 排序结果 |
|---|---|
| A 中位数填补（含延迟/成本） | DeepSeek V4.1 Flash > GLM-5.3-Flash > Qwen3.8-Flash-Next > GLM-5.3 > DeepSeek V4 Pro > **Kimi K3** > **Qwen3.8-Max** |
| B 可用维度重归一化 | DeepSeek V4.1 Flash > DeepSeek V4 Pro > Qwen3.8-Flash-Next > Kimi K3 > GLM-5.3 > GLM-5.3-Flash > Qwen3.8-Max |
| C 最差情况填补（min） | DeepSeek V4.1 Flash > GLM-5.3-Flash > GLM-5.3 > Qwen3.8-Flash-Next > Qwen3.8-Max > DeepSeek V4 Pro > **Kimi K3** |
| D 中位数填补、去掉延迟/成本 | **GLM-5.3** > **Kimi K3** > GLM-5.3-Flash > DeepSeek V4.1 Flash > Qwen3.8-Flash-Next > Qwen3.8-Max > DeepSeek V4 Pro |
| E 可用维度、去掉延迟/成本 | **Kimi K3** > DeepSeek V4.1 Flash > DeepSeek V4 Pro > GLM-5.3 > Qwen3.8-Flash-Next > Qwen3.8-Max > GLM-5.3-Flash |

**Kimi K3 在最差填补下垫底、在去掉延迟维度后排第一**——纯粹因为它 $15/M 的输出价与 44.8 tok/s 的速度。**Qwen3.8-Max 在 5 种策略中有 4 种位于末两位**，而它在 AA 指数与中文/智能体维度上都是五家第一。

> 这组数字不是本报告的缺陷，而是**本报告最重要的发现**：在当前公开证据下，**"五家厂商的单一能力总排名"在方法论上不可建立**。任何给出 1–7 精确名次且不附敏感性分析的报告，都在输出伪精度。

### 4.5 明确"我们不知道什么"

1. **网关别名对应的真实模型未知。** `qwen`（默认别名）、`kimi-explore`、`kimi-math`、`glm-zero-preview` 在公开评测中**没有任何对应条目**。所有引用这些别名的分数都只能标注为"该厂商家族的最优情形"。
2. **`deepseek-chat` / `deepseek-reasoner` 很可能仍在 V3.2 档位**，而所有 2026 年的公开评测测的是 V4-Pro / V4-Flash / V4.1-Flash。这是最大的版本错位风险。
3. **豆包全系无第三方可用数据**：无 LMArena Elo、无 LiveBench、无 RULER/MRCR、无 BFCL、无 IFEval、无实测 AA 指数。其唯一 AA 条目还是"估算值"。
4. **指令遵循维度出现测量真空**：本代旗舰（Qwen3.8-Max、GLM-5.3、Kimi K3、DeepSeek V4.x）在 IFBench 上**均无数据**；可得的只有上一代（Qwen3.7 Max 80.5、Kimi K2.6 76.0、GLM-5.2 73.3）。
5. **稳定性/可用性无任何第三方测量**：OpenRouter 提供 uptime 与延迟分位，但本报告未能取得五大厂商的逐节点可用性数据；D8 因此完全无法评分。这对本网关尤其关键（风控可用性与吞吐稳定性是历史痛点），却是最缺数据的一维。
6. **FrontierMath、ARC-AGI-2 上五家厂商全部无公开条目**——两个最能区分"真推理 vs 记忆"的仪器都用不上。
7. **Kimi K3 与 Qwen3.8-Max 在编码维度上存在方向性矛盾**：AA 的 Terminal-Bench 2.1 子分说 Qwen 更高（88.8 vs 85.0），官方 Terminal-Bench 4.0 榜说 GLM 远高（41.82 vs 26.97 vs Kimi 无条目）。**两版 Terminal-Bench 的难度差异是主因**，但无法在现有公开材料下调和。
8. **版本别名风险（version aliasing）**：闭源 Web 模型可能在不改别名的情况下静默替换权重。本报告所有分数都绑定到"抓取日期 + 模型版本号"，**任何一次静默替换都会使本报告失效**，且不会有任何公开通知。

---

## 5. 注意事项（Caveats）

1. **本报告的全部结论都是临时的。** 闭源 Web 模型的版本可在无通知下变更；五家厂商中只有智谱、月之暗面、DeepSeek 公布开源权重（仍不等于公开训练/评测流水线），阿里与字节的旗舰为闭源。**没有任何一家提供可复现的评测脚本与数据**。
2. **不存在"客观的能力总分"。** 复合分是**选择**的产物——归一化方式、权重、缺失值处理、聚合算子（均值 vs 最弱环节）都会改变名次。本报告已用真实数据证明：仅更换缺失值策略就能把 Kimi K3 从末位送到首位。
3. **厂商自报必须打折。** 三家厂商的技术报告/发布博客中的数字普遍高于第三方实测（例：DeepSeek V4-Pro 自报 SWE-bench Verified 80.6，独立机构测为 73.2–74；豆包自报体系与独立聚合器相差一倍以上）。本报告已对全部自报数字加了 **[V]** 标签并排除出评分。
4. **低质量聚合器泛滥。** 检索过程中出现大量 AI 生成或 SEO 内容农场（各类 "bench" 榜单站、镜像站），它们互相转载且数字彼此矛盾（同一模型的 LiveBench 分数在不同站点有 48.1 / 61.6 / 67.9 三个值）。**本报告尽量回溯到官方 CSV / 官方榜单页（livebench.ai、artificialanalysis.ai、tbench.ai、swebench.com），无法回溯的一律降为 [2ND] 并说明。**
5. **饱和基准不可用于排名。** AIME、MATH-500、GPQA Diamond、IFEval、NIAH 在 2026 年已饱和或被证明有污染，只能作回归看门狗。
6. **实时性衰减。** LiveBench 最新官方快照为 2026-06-25（已近 4 个月）；SWE-bench 官方榜最新条目停留在 2026-02。引用这些数字时必须同时引用快照日期。
7. **本报告未做任何实测。** 所有数字均来自公开材料。对"某个别名现在实际跑的是哪个模型"这一最关键的工程问题，**本报告无法回答**——这需要网关侧的可观测性（响应指纹、版本水印、输出特征比对）来补足。

---

## 6. 给本网关的可操作建议

1. **把别名升级为显式版本**：将 `qwen`、`kimi`、`deepseek-chat` 等模糊别名替换为带版本号的别名，并在网关日志中记录上游返回的模型标识（若可得）。
2. **按维度路由，而不是按"总排名"路由**：
   - 长上下文 / 深度推理 → **Kimi K3**（AA-LCR 88.7、HLE 46.9）
   - 编码智能体 / CLI 任务 → **GLM-5.3**（TB4.0 41.82）
   - 中文与结构化任务 → **Qwen3.8-Max**
   - 高吞吐低成本批处理 → **DeepSeek V4.1 Flash**（187 tok/s、$0.3/$1.2）
3. **把"证据覆盖率"作为第一等公民**：对 D6（指令遵循）与 D8（稳定性）建立**自建**小型评测与可用性埋点——公开数据在这两维上是空的，而这两维对本网关的实际体验影响最大。
4. **对豆包保持"能力未知"假设**：在获得独立数据前，不应基于其自报数字把它排入任何梯队。

---

## Sources

**评测方法论与聚合器**
- Artificial Analysis — Intelligence Benchmarking Methodology（v4.3.2）: https://artificialanalysis.ai/methodology/intelligence-benchmarking （抓取 2026-10-14）
- Artificial Analysis — LLM Leaderboard（模型指数、价格、速度、TTFT、逐项子分）: https://artificialanalysis.ai/leaderboards/models （抓取 2026-10-14）
- Artificial Analysis — Qwen3.8 Max 模型页: https://artificialanalysis.ai/models/qwen3-8-max （抓取 2026-10-14）
- Artificial Analysis — Performance Benchmarking Methodology（v2.2.0）: https://artificialanalysis.ai/methodology/performance-benchmarking
- LMArena / Arena-Rank 官方实现（Bradley-Terry + CI）: https://github.com/lmarena/arena-rank
- Chatbot Arena 论文: https://arxiv.org/abs/2403.04132
- LMSYS — Style Control 博客（2024-08-28）: https://www.lmsys.org/blog/2024-08-28-style-control/
- "The Leaderboard Illusion": https://arxiv.org/abs/2504.20879
- Stanford HELM（Capabilities / mean win rate）: https://crfm.stanford.edu/helm/capabilities/latest/ · https://arxiv.org/abs/2211.09110
- Epoch AI — Benchmarking: https://epoch.ai/benchmarks/about · ECI 文档: https://epoch.ai/data/eci-documentation/data
- Epoch AI — FrontierMath: https://epoch.ai/benchmarks/frontiermath （抓取 2026-10-14，未发现目标厂商条目）
- ARC Prize 排行榜: https://arcprize.org/leaderboard （抓取 2026-10-14，未发现目标厂商条目）
- OpenRouter — Rankings（用量份额 + 内嵌 AA 指数）: https://openrouter.ai/rankings （抓取 2026-10-14）
- OpenRouter — 供应商性能评测方法: https://openrouter.ai/blog/insights/evaluate-llm-provider-performance/
- JRC/OECD — Handbook on Constructing Composite Indicators（Step 4 归一化）: https://knowledge4policy.ec.europa.eu/sites/default/files/JRC_Week_2025_Step4.pdf
- "AI Cartography"（复合分测量误差与伪重复）: https://arxiv.org/abs/2605.25272
- "Balance of Benchmarks": https://arxiv.org/abs/2608.30044
- 污染与改写样本: https://arxiv.org/abs/2311.04850
- GEM 2026 污染综述: https://aclanthology.org/2026.gem-main.pdf
- IFBench: https://arxiv.org/abs/2507.02833
- NoLiMa: https://arxiv.org/abs/2502.05167
- 排名稳健性（少量偏好可改变榜首）: https://arxiv.org/abs/2508.11847
- Rank Intervals for Leaderboards: https://arxiv.org/html/2606.08679v1

**编码智能体基准**
- SWE-bench 官方排行榜（含 Verified / Multilingual / bash-only / lite 分榜）: https://www.swebench.com/ （抓取 2026-10-14；最新条目 2026-02）
- OpenAI — 为何不再报告 SWE-bench Verified: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified
- SWE-bench Pro（Scale AI）: https://scale.com/labs/swe-bench-pro-v2 · https://arxiv.org/abs/2509.16941
- SWE-Lancer: https://arxiv.org/abs/2502.12115
- Terminal-Bench 官方排行榜（4.0 / 2.1 版本）: https://www.tbench.ai/leaderboard （抓取 2026-10-14）
- LiveBench 官方数据（2026-06-25 快照 CSV）: https://livebench.ai/ · https://livebench.ai/v1/table_2026_06_25.csv · 方法论文: https://arxiv.org/abs/2406.19314
- LiveCodeBench: https://arxiv.org/abs/2403.07974

**LMArena 二手快照（lmarena.ai 网络不可达）**
- Prompeteer AI Model Leaderboard（数据源标注为 LMArena via Hugging Face，2026-10-08）: https://prompeteer.ai/leaderboard
- Arena 周榜第 41 周（2026-10-05~10-11）转述: https://news.17173.com/content/10122026/180731508.shtml · https://tech.ifeng.com/c/8xAGgGI3XAo
- LMArena 排行榜数据集（Hugging Face）: https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset

**中文评测**
- SuperCLUE 榜单汇总（二手）: https://www.datalearner.com/benchmarks/superclue
- SuperCLUE 2026-08 期次转述（国产模型格局）: https://www.toutiao.com/article/7677229124359913990/
- Chinese SimpleQA（ACL 2025，含 Doubao-pro-32k 与 GLM-4-Plus 成绩）: https://www.sohu.com/a/828836683_610300

**厂商材料（自报，仅作对照，未参与评分）**
- 字节 Seed 1.5 Pro: https://seed.bytedance.com/zh/special/doubao_1_5_pro
- 字节 Seed 1.6: https://seed.bytedance.com/zh/seed1_6
- 字节 Seed 1.8 模型卡: https://arxiv.org/pdf/2509.18713
- 豆包 Seed 2.1 Pro 发布（自报对标数据）: https://tech.ifeng.com/c/8uBalPRxcZq · https://post.smzdm.com/p/a036vzow/
- 豆包 Seed 2.1 系列百科: https://baike.baidu.com/item/%E8%B1%86%E5%8C%85%20Seed%202.1%E7%B3%BB%E5%88%97/68084170
- Kimi K2 Thinking 模型卡: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- Kimi K2.5 技术报告: https://arxiv.org/html/2602.02276v1
- Kimi k1.5 技术报告: https://ar5iv.labs.arxiv.org/html/2501.12599
- GLM-4.5 技术报告: https://ar5iv.labs.arxiv.org/html/2508.06471
- GLM-4.6 发布博客: https://z.ai/blog/glm-4.6
- GLM-5 模型卡/博客: https://huggingface.co/blog/mlabonne/glm-5
- GLM-5.3 文档: https://docs.z.ai/guides/llm/glm-5.3
- Qwen3.8-Max 更新公告（阿里云 Model Studio）: https://www.alibabacloud.com/en/notice/model_studio_update_notice_for_qwen38max_models_863
- DeepSeek API 更新与定价: https://api-docs.deepseek.com/updates · https://api-docs.deepseek.com/quick_start/pricing/

**低可信度来源（已降权，仅在无主源时作为线索，正文中以 [2ND] 标注）**
- RankLLMs（豆包 Seed 2.1 Pro 聚合分数，与厂商自报严重冲突）: https://rankllms.com/models/seed-2-1-pro/
- BenchGecko / metatext / ZerxLab / llmrun / swfte / toolhalla 等聚合站（数字互相转载且彼此矛盾，未用于评分）
