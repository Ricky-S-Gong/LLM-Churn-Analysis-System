# LLM 客户流失分析系统

基于 **RAG + QLoRA 微调** 的端到端电信客户流失分析系统，完全使用开源大语言模型驱动。在 Google Colab 免费 GPU 上即可完整运行，零 API 成本。

给定一条自然语言查询（如 *"光纤用户为什么流失？"*），系统会检索相关客户反馈，生成根本原因分析，评估流失风险，并返回带引用的可执行建议 -- 全部以结构化 JSON 格式输出。

本项目改编自 Ricky 在 Eth Tech 实习期间完成的一个 LLM 客户流失分析系统。由于企业数据受合规限制无法公开分享，我们使用了一个相似的 Kaggle 数据集进行简化复现。项目完全使用开源 Qwen 模型，支持完全可复现的零成本运行。

---

## 核心亮点

- **混合检索** -- FAISS（语义检索）+ BM25（关键词检索）+ 倒数排名融合（RRF）
- **开源大模型** -- Qwen2.5-7B-Instruct，通过 BitsAndBytes 进行 4-bit 量化
- **QLoRA 微调** -- 教师-学生蒸馏（Qwen2.5-14B -> Qwen2.5-7B），使用 PEFT + TRL
- **改进管线** -- 引用后处理验证 + 确定性风险评分，无需重新训练
- **结构化输出** -- JSON 格式，包含摘要、主要原因、风险等级、行动建议和引用
- **零成本** -- 完全在 Google Colab 免费额度上运行（T4 GPU）。建议开通 Colab Pro（A100/H100），可大幅加速训练数据生成和推理过程

---

## 架构概述

系统采用三阶段架构：

**阶段一 -- RAG 检索增强生成（Phase 1-4）：**
原始 CSV 数据经过加载、清洗后转换为统一的客户文档。每个文档通过 `BAAI/bge-base-en-v1.5`（768 维）生成向量嵌入并存入 FAISS 索引。查询时，混合检索（FAISS + BM25 + RRF）获取最相关的客户反馈，注入 Prompt 模板后发送至 Qwen2.5-7B-Instruct 生成结构化 JSON 响应。

**阶段二 -- QLoRA 微调（Phase 5-8）：**
教师模型（Qwen2.5-14B，4-bit 量化）通过 RAG 管线在 8 类查询模板上使用 k={3,5,7} 检索上下文生成 305 条高质量训练样本。学生模型（Qwen2.5-7B）使用 QLoRA（LoRA r=16，4000 万可训练参数，T4 上约 30-50 分钟）进行微调。基座模型与微调模型在 5 个质量指标上进行对比评估。

**阶段三 -- 改进管线（Final Demo）：**
后处理修复 Phase 8 评估中发现的两个回归问题 -- 无需重新训练：
1. **引用后处理**：验证 LLM 生成的引用是否在检索文档集中，移除幻觉 ID，并从检索集中补充。
2. **确定性风险评分**：用基于流失率、在网时长、月费和合同类型的加权数学公式替代 LLM 的主观风险判断。

改进管线在三方对比中实现了 **100% JSON 合规率**、**100% 引用准确率** 和 **92% 综合得分**。

---

## 阶段划分

| 阶段 | 名称 | 描述 |
|------|------|------|
| **1** | 数据准备 | 加载、清洗并合并电信流失数据集与客户反馈文本 |
| **2** | 索引与检索 | 生成 BGE 嵌入，构建 FAISS 索引，实现 BM25 + RRF 混合检索 |
| **3** | LLM 集成 | 加载 Qwen2.5-7B-Instruct（4-bit），设计 Prompt 模板，构建 RAG 查询管线 |
| **4** | 输出与评估 | 解析并验证结构化 JSON 输出，校验引用，评估检索和生成质量 |
| **5** | 训练数据生成 | 设计 8 类查询模板（122 条唯一查询），使用 Qwen2.5-14B 教师模型生成 305+54 条训练/验证样本 |
| **6** | QLoRA 微调 | 对 Qwen2.5-7B 进行 LoRA 微调（r=16，alpha=32，~4000 万可训练参数，T4 上约 30-50 分钟） |
| **7** | 模型集成 | 将微调模型（基座 + LoRA Adapter）重新接入 RAG 管线 |
| **8** | 评估与对比 | 在 5 个指标上对比基座模型与微调模型（JSON 合规、字段完整、类型正确、引用准确、风险一致） |
| **9** | 改进管线 | 通过后处理和确定性评分修复引用回归和风险不稳定问题（Final Demo） |

---

## 实验结果

### Phase 8：基座模型 vs 微调模型

| 指标 | 基座模型 | 微调模型 | 变化 |
|------|---------|---------|------|
| JSON 格式合规率 | 100.0% | 100.0% | +0.0% |
| 字段完整度 | 100.0% | 100.0% | +0.0% |
| 类型正确性 | 100.0% | 100.0% | +0.0% |
| 引用准确率 | 85.0% | 70.0% | **-15.0%** |
| 风险等级一致性 | 60.0% | 60.0% | +0.0% |
| **综合得分** | **89.0%** | **86.0%** | -3.0% |

### 三方对比（含改进管线）

| 指标 | 基座（原始） | 微调（原始） | 微调（改进） |
|------|------------|------------|------------|
| JSON 合规 | 100.0% | 100.0% | 100.0% |
| 字段完整 | 100.0% | 100.0% | 100.0% |
| 类型正确 | 100.0% | 100.0% | 100.0% |
| 引用准确 | 70.0% | 85.0% | **100.0%** |
| 风险一致 | 60.0% | 70.0% | 60.0% |
| **综合** | **86.0%** | **91.0%** | **92.0%** |

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.9+ |
| 数据处理 | Pandas, NumPy, scikit-learn |
| 向量嵌入 | `BAAI/bge-base-en-v1.5`（sentence-transformers） |
| 向量存储 | FAISS (IndexFlatIP) |
| 关键词检索 | rank-bm25 |
| LLM（RAG） | Qwen2.5-7B-Instruct（4-bit，BitsAndBytes） |
| LLM（教师模型） | Qwen2.5-14B-Instruct（4-bit，BitsAndBytes） |
| 微调框架 | PEFT (QLoRA), TRL (SFTTrainer), BitsAndBytes |
| 开发环境 | Jupyter Notebook, Google Colab (T4/A100 GPU) |

---

## 快速开始

### 本地运行（仅需 CPU）

```bash
git clone https://github.com/Ricky-S-Gong/LLM-Churn-Analysis-System.git
cd LLM-Churn-Analysis-System
pip install -r requirements.txt
```

### Google Colab 运行（需要 GPU）

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 上传 `notebooks/LLM-Churn-RAG-Fintuning.ipynb`（中文版）或 `notebooks/LLM-Churn-RAG-Finetuning-EN.ipynb`（英文版）
3. 设置运行时为 **T4 GPU**：运行时 -> 更改运行时类型 -> T4 GPU
4. 将 `data/telco_churn_with_all_feedback.csv` 上传至 Google Drive 并在 Colab 中挂载（推荐），或直接上传至 Colab 运行时
5. 从 Phase 1 开始依次运行

### 运行 Final Demo

1. 完成 Phase 1-8（或使用 Google Drive 上保存的 Adapter 权重）
2. 在 Colab 上使用 A100 GPU 打开 `notebooks/Final_Demo.ipynb`
3. 运行所有 Cell 查看改进管线及三方评估结果

### 编译论文

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## 显存规划（T4 -- 15 GB）

| 阶段 | 加载内容 | 预估显存 |
|------|---------|---------|
| Phase 5 | BGE 嵌入模型 (0.5 GB) + Qwen-14B 4-bit (8 GB) | ~9 GB |
| Phase 6 | Qwen-7B 4-bit (4.5 GB) + LoRA 层 + 优化器状态 | ~12 GB |
| Phase 7-8 | BGE 嵌入模型 (0.5 GB) + Qwen-7B 4-bit + LoRA Adapter (4.5 GB) | ~5 GB |

> 教师模型（Phase 5）在 Phase 6 微调开始前会被显式释放显存。

---

## 数据集

**Telco Customer Churn -- Realistic Customer Feedback**
- 来源：[Kaggle](https://www.kaggle.com/datasets/beatafaron/telco-customer-churn-realistic-customer-feedback/data)
- 内容：结构化流失数据（人口统计、服务订阅、账单信息、流失标签）+ 非结构化客户反馈文本
- 7,043 位客户，21 个特征，26.5% 流失率
- 主要文件：`data/telco_churn_with_all_feedback.csv`（5.4 MB）

---

## 文件结构

```
LLM-Churn-Analysis-System/
├── README.md                              # 英文版说明
├── README-Chinese.md                      # 本文件（中文版）
├── requirements.txt                       # Python 依赖
├── notebooks/
│   ├── LLM-Churn-RAG-Fintuning.ipynb     # 主代码文件（中文版）-- Phase 1-8
│   ├── LLM-Churn-RAG-Finetuning-EN.ipynb # 主代码文件（英文版）-- Phase 1-8
│   ├── Final_Demo.ipynb                   # 改进管线演示 -- Phase 9
│   └── Using_tool_required_for_customer_service.ipynb  # 工具调用参考示例
├── paper/
│   ├── paper.tex                          # 学术论文（NeurIPS 风格 LaTeX）
│   └── paper.pdf                          # 编译后的 PDF（16 页）
├── data/
│   ├── telco_churn_with_all_feedback.csv  # 主要数据集（5.4 MB）
│   ├── telco_noisy_feedback_prep.csv      # 预处理数据
│   └── telco_prep.csv                     # 预处理数据
├── lora_finetune_data/
│   ├── finetune_train_hf.jsonl            # 训练数据（305 条，ChatML 格式）
│   └── finetune_val_hf.jsonl              # 验证数据（54 条，ChatML 格式）
└── qwen2.5-7b-churn-lora/
    ├── final/                             # 最终 Adapter 权重（~77 MB）
    ├── checkpoint-50/                     # 中间检查点
    ├── checkpoint-100/                    # 中间检查点
    └── checkpoint-117/                    # 最终步检查点
```

---

## 项目进度

- [x] Phase 1: 数据准备
- [x] Phase 2: 索引与检索
- [x] Phase 3: LLM 集成（Qwen2.5-7B-Instruct，Colab T4）
- [x] Phase 4: 输出与评估
- [x] Phase 5: 训练数据生成（Colab T4）
- [x] Phase 6: QLoRA 微调（Colab T4）
- [x] Phase 7: 模型集成（Colab T4）
- [x] Phase 8: 评估与对比（Colab T4）
- [x] Phase 9: 改进管线（Final Demo，Colab A100）
- [x] 论文：学术论文（`paper.tex`）

---

## 预期输出示例

```json
{
  "summary": "对流失客户的分析表明，频繁的服务中断和不可靠的性能是促使客户离开的重要因素。",
  "top_reasons": [
    "频繁的服务中断",
    "互联网性能不稳定",
    "支付方式不便"
  ],
  "risk_level": "high",
  "actions": [
    "提高服务可靠性，减少中断",
    "增强客户支持响应能力",
    "提供更灵活便捷的支付选项"
  ],
  "citations": [
    "8065-YKXKD",
    "6892-EZDTG"
  ],
  "risk_score": 0.9014,
  "risk_components": {
    "churn_rate": 1.0,
    "tenure_risk": 0.9167,
    "charge_risk": 0.6112,
    "contract_risk": 1.0
  }
}
```

---

## 参考资源

- [Qwen2.5 模型系列](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PEFT -- 参数高效微调](https://huggingface.co/docs/peft)
- [TRL -- Transformer 强化学习库](https://huggingface.co/docs/trl)
- [BitsAndBytes 量化](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [BAAI/bge-base-en-v1.5 嵌入模型](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [FAISS -- Facebook AI 相似度搜索](https://github.com/facebookresearch/faiss)

---

## 许可

本项目仅用于教育和研究目的。
