<h1 align="center">PUMA</h1>

<h3 align="center">
  Know You Before You Speak:<br>
  User World Modeling for LLM Personalization in Multi-Turn Conversation
</h3>

<p align="center">
  <strong>面向动作选择的前瞻性用户世界建模</strong><br>
  理解潜在状态，预判交互走向，再决定如何回应。
</p>

<p align="center">
  <a href="https://github.com/Annie1161/PUMA"><img src="assets/badge-code.svg" alt="PUMA 代码"></a>
  <a href="LICENSE"><img src="assets/badge-license.svg" alt="MIT 许可证"></a>
  <a href="README.md"><img src="assets/badge-docs.svg" alt="English documentation"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#一研究动机">研究动机</a> ·
  <a href="#二方法概览">方法概览</a> ·
  <a href="#三实验结果">实验结果</a> ·
  <a href="#四安装与模型配置">安装配置</a> ·
  <a href="#五数据准备">数据准备</a> ·
  <a href="#六离线状态评估">离线评估</a> ·
  <a href="#七交互式动态评估">动态评估</a>
</p>

> **同一句话，在不同用户状态下，可能具有不同的含义。**
> 观测（observation）本身不能唯一、准确地揭示用户的真实状态。PUMA 持续维护对潜在状态的信念，
> 并通过用户世界模型预测不同回复可能带来的后续变化。

## 一、研究动机

我们能直接看到用户说了什么，却往往无法直接看到其不确定感、投入程度和行动准备度。
这些潜在因素只会部分地体现在语言中。如果将话语直接等同于真实状态，
系统就可能给出字面上相关、却没有回应用户实际处境的回复。

例如，用户说的都是：**“我已经服用这个新药两周了。”**

| 可能的潜在状态 | 这句话可能传达的含义 | 回复需要进一步了解什么 |
| --- | --- | --- |
| 担忧副作用 | “我对自己的感受有些不安，希望获得回应。” | 邀请用户表达具体顾虑。 |
| 有信心，正在记录进展 | “我在向你更新最近的情况。” | 询问进展和使用体验。 |
| 对是否继续感到犹豫 | “我正在重新考虑是否继续。” | 澄清犹豫的原因和用户的意图。 |

*以上为根据论文动机示例整理的可能解释。*

仅凭这句话，系统无法确定哪一种解释更合适。历史对话和新的观测提供推断依据，
进一步提问也可以帮助降低不确定性。**PUMA 正是为解决这种歧义而设计：将状态推断与动作选择连接起来，
跟踪可能的用户状态，并在生成回复前预测不同对话动作的后果。**

## 二、方法概览

PUMA 以自由能原理（Free Energy Principle，FEP）为基础，
将个性化对话建模为面向**部分可观测、持续变化的用户**的序贯决策。
用户世界模型将潜在状态、可观测话语和由系统动作影响的状态转移联系起来。

<p align="center">
  <a href="assets/puma-framework.png"><img src="assets/puma-framework.png" width="100%" alt="PUMA 方法框架：用户状态信念与世界模型、期望自由能动作选择、回复生成"></a>
</p>

<p align="center"><sub>从观察用户话语，到理解可能的潜在状态，再预判后续交互。</sub></p>

| 组成部分 | 在交互中的作用 |
| --- | --- |
| **状态信念** `q(s)` | 维护用户当前可能处于哪些状态及其概率，并随新证据更新。 |
| **观测模型** `p(o \| s)` | 描述不同潜在状态可能如何表达为可观测的话语。 |
| **转移模型** `p(s′ \| s, a)` | 预测候选系统动作可能如何改变用户状态。 |
| **前瞻性动作选择** | 近似计算期望自由能（EFE），平衡降低不确定性与推动目标进展。 |
| **记忆与回复生成** | 结合用户上下文，将选定动作转化为自然语言回复。 |

每轮交互形成一个循环：**观测话语 → 更新状态信念与世界模型 → 评估候选动作 → 生成回复 → 接收新观测**。
状态信念始终是可以随证据修正的估计。

当前实现将这一框架用于动机式访谈（Motivational Interviewing，MI），
以**前意向期 → 意向期 → 准备期**表示改变准备度。
离线评估还支持 **neutral / change / sustain** 言语标签。
世界模型的更新通过交互中的 LLM 推理维护结构化观测规则和转移规则；当前提供的是推理与评估流程。

## 三、实验结果

配套论文分别通过固定对话评估状态跟踪能力，通过基于画像的患者模拟器评估多轮交互表现。
以下图表和数值均来自该论文。

<table>
  <tr>
    <td align="center" width="50%"><strong>跟踪潜在状态</strong></td>
    <td align="center" width="50%"><strong>预判用户变化</strong></td>
  </tr>
  <tr>
    <td><a href="assets/state-prediction.png"><img src="assets/state-prediction.png" width="100%" alt="不同对话阶段的下一状态预测准确率：PUMA soft fusion 与长提示词方法"></a></td>
    <td><a href="assets/readiness-progression.png"><img src="assets/readiness-progression.png" width="100%" alt="CAMI 模拟交互中 PUMA 与对比方法的准备度变化曲线"></a></td>
  </tr>
</table>

左图展示不同对话阶段的下一状态预测准确率；右图展示 CAMI 模拟交互中的准备度变化。
右图原始图例中的 **FEP (Ours)** 对应 PUMA。

| 基座模型 | 状态提升 ↑ | 到达准备期比例 ↑ | 触发条件覆盖率 ↑ | 平均轮数 ↓ |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B | 1.62 | 75.9% | 62.4% | 12.2 |
| Llama-3.1-8B | 1.76 | 83.0% | 63.2% | 10.7 |

*论文所报告的 PUMA 结果，评估范围为其 CAMI 患者模拟器协议。*

## 四、安装与模型配置

### 准备环境

以下命令适用于 Bash，后续示例均从项目根目录执行。

获取项目并创建虚拟环境：

```bash
git clone https://github.com/Annie1161/PUMA.git
cd PUMA
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

requirements.txt 是仓库附带的环境依赖快照，其中固定了 torch==1.13.0+cu117
和 transformers==4.37.0。该组合尚未验证可运行 Qwen3。
使用下方 Qwen3 示例前，需要准备与模型兼容的 PyTorch、Transformers 和 CUDA 环境，
并协调依赖快照中的版本；带 CUDA 后缀的安装包还可能需要对应的软件包索引。

若使用依赖快照对应的环境，安装命令为：

```bash
python -m pip install -r requirements.txt
```

记忆模块还需要 spaCy 英文模型：

```bash
python -m spacy download en_core_web_sm
```

设置当前终端的模块搜索路径：

```bash
export PYTHONPATH="$PWD/src:$PWD/src/agent${PYTHONPATH:+:$PYTHONPATH}"
```

### 选择本地模型

所有模型调用均使用本地 Hugging Face 模型。通过 --model 指定模型名称或本地模型目录。
智能体默认模型为 Qwen/Qwen3-0.6B，动态评估默认模型为 Qwen/Qwen3-8B。
智能体的默认模型也可通过 PUMA_MODEL 环境变量覆盖。

最小本地推理逻辑位于 src/agent/counselor.py。动态评估中，咨询师和患者模拟器
复用同一个已加载的模型。

## 五、数据准备

数据集需要自行准备，并遵守原始数据的获取和许可条件。
下方命令假定数据放置在以下位置：

```text
data/cami/profiles.jsonl
data/cami/profiles_misc.jsonl
```

CAMI 画像文件采用 JSONL 格式，每行一个 JSON 对象。
对话字段包括 speakers、utterances、states、talk_types、actions。
画像字段包括 Personas、Beliefs、Motivation、Acceptable Plans。
人群统计还使用 misc_actions 字段记录咨询师的 MISC 动作标签。
同一对话中各轮次字段的数组应保持对齐。

请自行准备上述格式的对话数据和已标注的画像文件。

运行时请通过命令行显式指定数据路径和评估参数。

## 六、离线状态评估

在准备好模型环境和数据后，对固定 CAMI 对话执行状态预测：

```bash
python src/agent/main.py \
    --dataset cami \
    --data_path data/cami \
    --data_name profiles.jsonl \
    --model Qwen/Qwen3-8B \
    --state_label_space cami_stage \
    --persona_state_space cami_stage \
    --observation_focused_state True \
    --prior_fusion_mode soft_fusion \
    --test_num 5 \
    --results_file logs/results/puma_static.json
```

## 七、交互式动态评估

对 5 个画像执行多轮咨询，每个画像最多运行 20 轮。示例显式开启 EFE 动作选择，
并通过 soft fusion 融合观测推断结果与规划器预测的状态先验：

```bash
python src/tools/eval_simple_sim.py \
    --profiles_misc_path data/cami/profiles_misc.jsonl \
    --model Qwen/Qwen3-8B \
    --test_num 5 \
    --max_turns 20 \
    --seed 42 \
    --fep_efe_action True \
    --prior_fusion_mode soft_fusion \
    --output_json logs/results/simple_sim_puma_n5.json
```

结果包含汇总指标和逐画像轨迹，包括状态提升、到达准备期的比例、对话轮数、
触发条件覆盖率和动作分布。

动态评估使用 PUMA 咨询师和本地患者模拟器。
