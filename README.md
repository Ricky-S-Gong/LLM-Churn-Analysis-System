# LLM Customer Churn Analysis System

An end-to-end **RAG + QLoRA fine-tuning** system for telecom customer churn analysis, powered entirely by open-source LLMs. Runs fully on a free Google Colab GPU with zero API cost.

Given a natural-language query (e.g. *"Why are fiber optic customers churning?"*), the system retrieves relevant customer feedback, generates root-cause analysis, assesses churn risk, and returns actionable recommendations with citations -- all in structured JSON.

This project is adapted from an LLM churn analysis system Ricky built during an Eth Tech internship. Because enterprise data cannot be shared due to compliance constraints, we use a similar Kaggle dataset for a simplified reproduction. The project uses fully open-source Qwen models, enabling reproducible, zero-cost execution.

---

## Key Features

- **Hybrid Retrieval** -- FAISS (semantic) + BM25 (keyword) with Reciprocal Rank Fusion (RRF)
- **Open-Source LLM** -- Qwen2.5-7B-Instruct, 4-bit quantized via BitsAndBytes
- **QLoRA Fine-Tuning** -- Teacher-student distillation (Qwen2.5-14B -> Qwen2.5-7B) using PEFT + TRL
- **Improved Pipeline** -- Post-processing for citation validation + deterministic risk scoring
- **Structured Output** -- JSON with summary, top reasons, risk level, actions, and citations
- **Academic Paper** -- Full NeurIPS-style LaTeX paper (`paper.tex`) with mathematical formulations
- **Zero Cost** -- Runs entirely on Google Colab free tier (T4/A100 GPU)

---

## Architecture Overview

The system follows a three-stage architecture:

**Stage 1 -- RAG Pipeline (Phase 1-4):**
Raw CSV data is ingested, cleaned, and converted into unified customer documents. Each document is embedded using `BAAI/bge-base-en-v1.5` (768-dim) and indexed in FAISS. At query time, hybrid retrieval (FAISS + BM25 + RRF) fetches the most relevant customer feedback, which is injected into a prompt template and sent to Qwen2.5-7B-Instruct for structured JSON generation.

**Stage 2 -- QLoRA Fine-Tuning (Phase 5-8):**
A teacher model (Qwen2.5-14B, 4-bit) generates 305 high-quality training samples via the RAG pipeline across 8 query categories with k={3,5,7} retrieval contexts. The student model (Qwen2.5-7B) is fine-tuned with QLoRA (LoRA r=16, 40M trainable params, ~30-50 min on T4). Base and fine-tuned models are evaluated side-by-side on 5 quality metrics.

**Stage 3 -- Improved Pipeline (Final Demo):**
Post-processing fixes address two regressions found in Phase 8 evaluation -- without retraining:
1. **Citation Post-Processing**: Validates LLM-generated citations against retrieved customer IDs, removes hallucinated IDs, and supplements from the retrieved set.
2. **Deterministic Risk Scoring**: Replaces subjective LLM risk judgment with a weighted mathematical formula over churn rate, tenure, monthly charges, and contract type.

The improved pipeline achieves **100% JSON compliance**, **100% citation accuracy**, and **92% overall score** in a 3-way comparison (base vs. fine-tuned vs. improved).

---

## Phase Breakdown

| Phase | Title | Description |
|-------|-------|-------------|
| **1** | Data Ingestion | Load, clean, and merge the Telco Churn dataset with customer feedback text |
| **2** | Indexing & Retrieval | Generate BGE embeddings, build FAISS index, implement BM25 + RRF hybrid retrieval |
| **3** | LLM Integration | Load Qwen2.5-7B-Instruct (4-bit), design prompt templates, build the RAG query pipeline |
| **4** | Output & Evaluation | Parse and validate structured JSON output, verify citations, evaluate retrieval and generation quality |
| **5** | Training Data Generation | Design 8 categories of query templates (122 unique queries), generate 305+54 training/validation samples using Qwen2.5-14B as teacher |
| **6** | QLoRA Fine-Tuning | Fine-tune Qwen2.5-7B with LoRA (r=16, alpha=32, ~40M trainable params, ~30-50 min on T4) |
| **7** | Model Integration | Load fine-tuned model (base + LoRA adapter) back into the RAG pipeline |
| **8** | Evaluation & Comparison | Compare base vs. fine-tuned model on 5 metrics (JSON valid, fields complete, types correct, citation accuracy, risk aligned) |
| **9** | Improved Pipeline | Fix citation regression and risk instability via post-processing and deterministic scoring (Final Demo) |

---

## Results

### Phase 8: Base vs. Fine-Tuned

| Metric | Base Model | Fine-Tuned | Change |
|--------|-----------|-----------|--------|
| JSON Format Compliance | 100.0% | 100.0% | +0.0% |
| Field Completeness | 100.0% | 100.0% | +0.0% |
| Type Correctness | 100.0% | 100.0% | +0.0% |
| Citation Accuracy | 85.0% | 70.0% | **-15.0%** |
| Risk Level Alignment | 60.0% | 60.0% | +0.0% |
| **Overall** | **89.0%** | **86.0%** | -3.0% |

### 3-Way Comparison (with Improved Pipeline)

| Metric | Base (original) | FT (original) | FT (improved) |
|--------|----------------|---------------|---------------|
| JSON Valid | 100.0% | 100.0% | 100.0% |
| Fields Complete | 100.0% | 100.0% | 100.0% |
| Types Correct | 100.0% | 100.0% | 100.0% |
| Citation Accuracy | 70.0% | 85.0% | **100.0%** |
| Risk Aligned | 60.0% | 70.0% | 60.0% |
| **Overall** | **86.0%** | **91.0%** | **92.0%** |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy, scikit-learn |
| Embeddings | `BAAI/bge-base-en-v1.5` via sentence-transformers |
| Vector Store | FAISS (IndexFlatIP) |
| Keyword Search | rank-bm25 |
| LLM (RAG) | Qwen2.5-7B-Instruct (4-bit, BitsAndBytes) |
| LLM (Teacher) | Qwen2.5-14B-Instruct (4-bit, BitsAndBytes) |
| Fine-Tuning | PEFT (QLoRA), TRL (SFTTrainer), BitsAndBytes |
| Environment | Jupyter Notebook, Google Colab (T4/A100 GPU) |

---

## Quick Start

### Local Setup (CPU only)

```bash
git clone https://github.com/Ricky-S-Gong/LLM-Churn-Analysis-System.git
cd LLM-Churn-Analysis-System
pip install -r requirements.txt
```

### Google Colab Setup (GPU required)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/LLM-Churn-RAG-Finetuning-EN.ipynb` (or the Chinese version `notebooks/LLM-Churn-RAG-Fintuning.ipynb`)
3. Set runtime to **T4 GPU**: Runtime -> Change runtime type -> T4 GPU
4. Upload `data/telco_churn_with_all_feedback.csv` to Colab (or mount Google Drive)
5. Run cells sequentially from Phase 1

### Run the Final Demo

1. Complete Phase 1-8 (or use saved adapter weights from Google Drive)
2. Open `notebooks/Final_Demo.ipynb` on Colab with A100 GPU
3. Run all cells to see the improved pipeline with 3-way evaluation

### Compile the Paper

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## VRAM Planning (T4 -- 15 GB)

| Phase | Components Loaded | Est. VRAM |
|-------|-------------------|-----------|
| Phase 5 | BGE embeddings (0.5 GB) + Qwen-14B 4-bit (8 GB) | ~9 GB |
| Phase 6 | Qwen-7B 4-bit (4.5 GB) + LoRA layers + optimizer states | ~12 GB |
| Phase 7-8 | BGE embeddings (0.5 GB) + Qwen-7B 4-bit + LoRA adapter (4.5 GB) | ~5 GB |

> The teacher model (Phase 5) is explicitly freed from VRAM before fine-tuning begins in Phase 6.

---

## Dataset

**Telco Customer Churn -- Realistic Customer Feedback**
- Source: [Kaggle](https://www.kaggle.com/datasets/beatafaron/telco-customer-churn-realistic-customer-feedback/data)
- Contents: Structured churn data (demographics, services, billing, churn label) + unstructured customer feedback text
- 7,043 customers, 21 features, 26.5% churn rate
- Primary file: `data/telco_churn_with_all_feedback.csv` (5.4 MB)

---

## File Structure

```
LLM-Churn-Analysis-System/
├── README.md                              # This file (English)
├── README-Chinese.md                      # Chinese version
├── requirements.txt                       # Python dependencies
├── notebooks/
│   ├── LLM-Churn-RAG-Fintuning.ipynb     # Main notebook (Chinese) -- Phase 1-8
│   ├── LLM-Churn-RAG-Finetuning-EN.ipynb # Main notebook (English) -- Phase 1-8
│   ├── Final_Demo.ipynb                   # Improved pipeline demo -- Phase 9
│   └── Using_tool_required_for_customer_service.ipynb  # Tool-use reference example
├── paper/
│   ├── paper.tex                          # Academic paper (NeurIPS-style LaTeX)
│   └── paper.pdf                          # Compiled PDF (16 pages)
├── data/
│   ├── telco_churn_with_all_feedback.csv  # Primary dataset (5.4 MB)
│   ├── telco_noisy_feedback_prep.csv      # Preprocessed data
│   └── telco_prep.csv                     # Preprocessed data
├── lora_finetune_data/
│   ├── finetune_train_hf.jsonl            # Training data (305 samples, ChatML)
│   └── finetune_val_hf.jsonl              # Validation data (54 samples, ChatML)
└── qwen2.5-7b-churn-lora/
    ├── final/                             # Final adapter weights (~77 MB)
    ├── checkpoint-50/                     # Intermediate checkpoint
    ├── checkpoint-100/                    # Intermediate checkpoint
    └── checkpoint-117/                    # Final step checkpoint
```

---

## Progress

- [x] Phase 1: Data Ingestion
- [x] Phase 2: Indexing & Retrieval
- [x] Phase 3: LLM Integration (Qwen2.5-7B-Instruct on Colab T4)
- [x] Phase 4: Output & Evaluation
- [x] Phase 5: Training Data Generation (Colab T4)
- [x] Phase 6: QLoRA Fine-Tuning (Colab T4)
- [x] Phase 7: Model Integration (Colab T4)
- [x] Phase 8: Evaluation & Comparison (Colab T4)
- [x] Phase 9: Improved Pipeline (Final Demo, Colab A100)
- [x] Paper: Academic paper (`paper.tex`)

---

## Expected Output Example

```json
{
  "summary": "The analysis of churned customers indicates that frequent service interruptions and unreliable performance are significant factors contributing to their decisions to leave.",
  "top_reasons": [
    "Frequent service outages",
    "Unreliable internet performance",
    "Inconvenience with payment methods"
  ],
  "risk_level": "high",
  "actions": [
    "Improve service reliability and reduce outages",
    "Enhance customer support responsiveness",
    "Offer more flexible and convenient payment options"
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

## References

- [Qwen2.5 Model Collection](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PEFT -- Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)
- [TRL -- Transformer Reinforcement Learning](https://huggingface.co/docs/trl)
- [BitsAndBytes Quantization](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [BAAI/bge-base-en-v1.5 Embedding Model](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [FAISS -- Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)

---

## License

This project is for educational and research purposes.
