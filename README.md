# 🧾 Advanced Structured Invoice Extraction: LLMs vs. VLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Qwen](https://img.shields.io/badge/Model-Qwen2--VL--7B-purple)](https://huggingface.co/Qwen)
[![Florence-2](https://img.shields.io/badge/Model-Florence--2--Base-cyan)](https://huggingface.co/microsoft/Florence-2-base)

An enterprise-grade evaluation pipeline designed to solve complex Key-Value (KV) pair extraction and line-item grouping on complex retail receipts using state-of-the-art Generative AI. 

Standard heuristic algorithms (like Regex and spatial clustering) become unmaintainable as receipt layouts scale in diversity. This repository abandons rigid rules in favor of cognitive extraction, benchmarking **Two-Stage Spatial LLMs** against **End-to-End Vision-Language Models (VLMs)**, and exploring custom fine-tuning for extreme robustness.

## 🏗️ Architectures Evaluated

1. **Phase 1: Two-Stage Spatial LLM Pipeline**
   * **Detector:** PaddleOCR (Extracts text and `[x, y]` pixel coordinates).
   * **Serializer:** A custom Python function that sorts and serializes the bounding boxes into a spatial string (e.g., `[x:120, y:45] Total $4.99`).
   * **Brain:** `Qwen-2.5-7B-Instruct` (4-bit quantized) using strict prompt engineering to force JSON generation.
2. **Phase 2: End-to-End Vision-Language Model (VLM)**
   * **Model:** `Qwen2-VL-7B-Instruct` (4-bit quantized).
   * **Method:** Bypasses external OCR entirely. Natively processes the raw image pixels alongside the system prompt to output structured JSON in a single forward pass.
3. **Fine-Tuned Multimodal Extraction (Florence-2)**
   * **Model:** Microsoft's `Florence-2-base`.
   * **Method:** Custom fine-tuning loop (`AutoModelForCausalLM`, dynamic padding, custom collators) designed to train the model to extract flat-entity KV pairs from heavily degraded/blurred documents.

---

## 🏆 Benchmark Results

Evaluated on the **CORD-v2** dataset for strict JSON schema compliance. 

### 1. Generative Extraction Accuracy
*Note: VLM inference was capped at 800x800 resolution to prevent OOM errors on the Kaggle T4 GPU.*

| Pipeline Strategy | Model | Total Price Acc | Item Count Acc |
| :--- | :--- | :---: | :---: |
| **Phase 1: Serialized OCR + LLM** | Qwen-2.5-7B | 34.0% | **46.0%** |
| **Phase 2: End-to-End VLM** | Qwen2-VL-7B | **58.0%** | 44.0% |

### 2. Florence-2 Robustness Stress Test (Heavy Blur)
We fine-tuned `Florence-2-base` specifically to test its behavior under catastrophic image degradation (heavy Gaussian blur simulating extreme motion blur). We implemented a custom fuzzy-matching sequence evaluator to track precision vs. recall.

| Metric | Score | Insight |
| :--- | :---: | :--- |
| **Precision** | **61.5%** | When the model makes a prediction, it is usually correct. |
| **Recall** | 27.7% | The model struggles to find all fields in heavy blur. |
| **F1 Score** | 38.2% | Overall harmonic mean under extreme stress. |

---

## 💡 Key Architectural Insights

1. **The OCR Bottleneck:** The Phase 1 pipeline proved that text-only LLMs are exceptional at parsing meaning, but they are severely bottlenecked by the traditional OCR engine. If PaddleOCR misses a decimal point or misaligns a bounding box, the LLM hallucinates to compensate. 
2. **VLM Resolution vs. VRAM Limits:** In Phase 2, `Qwen2-VL` dominated the Total Price extraction (58% vs 34%). However, its Item Count Accuracy dipped to 44%. Because Kaggle's T4 GPU limits forced us to aggressively downscale massive, vertical receipts to an 800x800 grid, the VLM simply could not visually resolve the micro-print used for line items.
3. **The "Conservative" Model Strategy:** Our Florence-2 stress test revealed a fascinating model behavior. When blinded by severe blur, the fine-tuned VLM adopted a highly conservative extraction strategy. Rather than hallucinating incorrect values to fill the JSON, the model safely dropped unreadable fields. This resulted in a low recall (27%) but successfully preserved a strong precision (61%) for the data it could securely resolve.

---

## 🚀 Production Deployment Blueprint

To push this prototype from 60% to 95%+ enterprise-grade accuracy, this repository establishes the exact deployment roadmap:

* **The Hardware Fix (Remove the Cap):** Deploy `Qwen2-VL-7B` on a larger GPU (e.g., NVIDIA A10G with 24GB VRAM). This allows the image to be passed at `2000x2000` resolution. Once the VLM's vision encoder can actually read the fine print, the Item Count Accuracy will surge.
* **The Software Fix (Constrained Decoding):** Instead of writing regex scripts to clean up markdown blocks or hoping the VLM doesn't miss a comma, the production pipeline should wrap the VLM in a library like **Outlines** or **vLLM**. This forces the model at the hardware/logit level to only generate tokens that perfectly match the required JSON schema.

## 💻 Reproducibility & Quick Start

### 1. Environment Setup
```bash
git clone [https://github.com/ocr-workspace/Retail-printed-reciept-ocr-test.git](https://github.com/ocr-workspace/Retail-printed-reciept-ocr-test.git)
cd Retail-printed-reciept-ocr-test

# Install heavy VLM dependencies
pip install transformers==4.49.0 accelerate bitsandbytes qwen-vl-utils evaluate
pip install paddlepaddle-gpu paddleocr

2. Run the Benchmark
# Execute the Jupyter Notebook
jupyter notebook Structured_Invoice_Vast.ipynb

🤝 Contributing
Contributions are welcome! If you have optimized the cluster_rows function to handle diagonally warped receipts, or have fine-tuned a LayoutLMv3 model to benchmark against Donut, please open a PR.
