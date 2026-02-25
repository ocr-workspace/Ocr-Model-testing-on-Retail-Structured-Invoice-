# 🧾 Retail Receipt OCR & Spatial Extraction Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/)
[![Dataset: CORD-v2](https://img.shields.io/badge/Dataset-CORD--v2-lightgrey)](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

An advanced evaluation pipeline designed to benchmark AI models on their ability to extract **structured data** (Key-Value pairs like `Total_Price` and `Item_Count`) from complex retail receipts. 

Extracting raw text from a receipt is a solved problem. Logically grouping that text into coherent line items, menus, and prices despite warped paper, dense tabular structures, and unpredictable vertical spacing is highly difficult. This repository contrasts traditional OCR + Spatial Clustering against modern OCR-free Vision-Language Models (VLMs).

## 🏗️ Architecture Evaluated

1. **Two-Stage OCR + Heuristics:**
   * **Extractors Evaluated:** PyTesseract, EasyOCR, PaddleOCR (v2.8), DocTR, and a custom Hybrid (Paddle Detection + Microsoft TrOCR Recognition).
   * **Post-Processing:** Regex and character mapping.
2. **Spatial Reconstruction Pipeline:**
   * Uses raw bounding box coordinates (`[x_min, y_min, x_max, y_max]`) from PaddleOCR.
   * Employs adaptive **Y-Coordinate Proximity Clustering** (inspired by DBSCAN) to logically reassemble scattered bounding boxes into unified horizontal rows.
3. **End-to-End Document Understanding (OCR-Free):**
   * **Donut** (*Document Understanding Transformer*): Bypasses text detection entirely, mapping raw image pixels directly to a structured JSON output using a Swin Transformer encoder and MBART decoder.

---

## 🏆 Benchmark Results

### 1. Raw Text Recognition Metrics
Evaluated on the **CORD-v2** dataset to measure raw string extraction capability.

| Model / Pipeline | CER (↓) | WER (↓) | Avg Time/Img |
| :--- | :---: | :---: | :---: |
| **PaddleOCR** | **1.08** | **1.16** | **0.16s** |
| **EasyOCR** | 1.16 | 1.74 | 0.52s |
| **DocTR** | 1.19 | 1.40 | 2.05s |
| **Tesseract** | 1.24 | 1.66 | 0.90s |
| **Hybrid (Paddle + TrOCR)**| 1.25 | 1.38 | 1.93s |

### 2. Structured Extraction Accuracy
Did the model successfully identify the exact numeric value of the `Total Price` and the exact quantity of purchased `Items`?

| Pipeline Strategy | Total Price Acc | Item Count Acc |
| :--- | :---: | :---: |
| OCR + Basic Regex (Paddle) | 10.0% | 0.0% |
| Hybrid OCR + Regex | 20.0% | 0.0% |
| **OCR + Spatial Row Clustering** | **46.0%** | **14.0%** |
| **Donut VLM (End-to-End)** | **100.0%** | **100.0%** |

### 3. VLM Robustness Testing (Stress Test)
Donut achieved 100% accuracy on clean data. We subjected the dataset to OpenCV-based augmentations to measure the model's structural resilience to real-world camera degradation.

| Perturbation Type | Simulated Environment | Total Price Acc | Item Count Acc |
| :--- | :--- | :---: | :---: |
| **Rotate** (5 degrees) | Hasty camera angle | 100% | 100% |
| **Brightness** (alpha=1.2) | Harsh flash / glare | 98% | 98% |
| **Blur** (7x7 Gaussian) | Motion blur / out-of-focus | 88% | 88% |
| **Downscale** (50%) | Low-res camera / compression | 86% | 94% |
| **Gaussian Noise** | High ISO / dark room | **34%** | **40%** |

---

## 💡 Key Architectural Insights

1. **Regex is Dead for Receipts:** Relying on regular expressions (`\d+\s*x`) to parse raw, concatenated OCR strings failed completely (0% Item Count Accuracy across all standard models). 
2. **Spatial Math is Mandatory:** By dynamically grouping bounding boxes based on the image's height (`y_threshold = avg_gap * 1.2`), our Spatial Reconstruction algorithm boosted PaddleOCR's structured extraction accuracy by **360%** (10% → 46%).
3. **Donut Dominates but is Fragile:** Donut natively understands document topology, achieving a flawless 100% baseline. However, the Robustness tests prove that End-to-End vision transformers are highly sensitive to pixel-level degradation—introducing Gaussian noise caused catastrophic failure (accuracy dropped by 66%), whereas traditional OCR engines usually survive minor noise.

## 🚀 Reproducibility & Quick Start

### 1. Environment Setup
*Note: This repository requires locking NumPy to the 1.x series to prevent C++ binding crashes with legacy image augmentation libraries (imgaug) required by PaddleOCR.*

```bash
git clone [https://github.com/ocr-workspace/Retail-printed-reciept-ocr-test.git](https://github.com/ocr-workspace/Retail-printed-reciept-ocr-test.git)
cd Retail-printed-reciept-ocr-test

# Prevent C++ dependency crashes
pip install "numpy<2.0" 

# Install OCR and Evaluation Dependencies
pip install paddlepaddle-gpu==2.6.2 -f [https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html](https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html)
pip install paddleocr==2.8.1 python-doctr transformers evaluate easyocr jiwer

2. Run the Benchmark
jupyter notebook structured-table-invoice-ipynb.ipynb

🤝 Contributing
Contributions are welcome! If you have optimized the cluster_rows function to handle diagonally warped receipts, or have fine-tuned a LayoutLMv3 model to benchmark against Donut, please open a PR.
