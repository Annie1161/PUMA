# PUMA：面向动作选择的前瞻性用户状态建模

[English](README.md) | [简体中文](README.zh-CN.md)
项目地址：https://github.com/Annie1161/PUMA

## 一、项目介绍

PUMA 是面向动机式访谈（Motivational Interviewing，MI）对话的研究框架。
它根据对话和用户画像推断用户当前的改变准备度，预测不同咨询动作可能带来的状态变化，
并通过期望自由能（Expected Free Energy，EFE）指导动作选择。

系统将状态推断、动作规划和回复生成划分为独立模块，并通过对话记忆维护跨轮次信息。
每轮对话的基本流程为：

1. 根据用户画像和近期对话推断潜在状态的后验分布。
2. 使用转移模型预测候选咨询动作对后续状态的影响。
3. 通过 EFE 评分选择动作，并生成自然语言回复。
4. 更新对话记忆，将状态信息传递到下一轮。

准备度状态包括：前意向期（precontemplation）、意向期（contemplation）、
准备期（preparation）。离线评估还支持 neutral、change、sustain 三类言语标签。
观测聚焦推断和规划先验融合均可配置，实验时应明确指定相关参数。

## 二、主要功能

PUMA 智能体：用户画像建模、状态推断、转移规划、回复生成和对话记忆。
患者模拟器：基于画像生成回复，支持语义触发匹配和个体化动作分布。
评估工具：离线状态预测、多轮对话和模型裁判评估。
行为标注工具：MISC 咨询动作分类及数据标注。

## 三、环境准备

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

## 四、模型配置

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

## 七、使用患者模拟器进行动态评估

对 5 个画像执行多轮咨询，每个画像最多运行 20 轮：

```bash
python src/tools/eval_simple_sim.py \
    --profiles_misc_path data/cami/profiles_misc.jsonl \
    --model Qwen/Qwen3-8B \
    --test_num 5 \
    --max_turns 20 \
    --seed 42 \
    --prior_fusion_mode soft_fusion \
    --output_json logs/results/simple_sim_puma_n5.json
```

结果包含汇总指标和逐画像轨迹，包括状态提升、到达准备期的比例、对话轮数、
触发条件覆盖率和动作分布。

动态评估使用 PUMA 咨询师和本地患者模拟器。
