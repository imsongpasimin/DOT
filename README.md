# Dynamic Order Template (DOT) for Generative Aspect-Based Sentiment Analysis

> **ACL 2025 · Main Track**
> Official PyTorch implementation of **“Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis.”**

---

## 📌 Abstract
<p align="center">
  <img src="https://github.com/user-attachments/assets/979657a2-292d-4aa2-a111-c98195a185da" width="523" height="457" alt="image" />
</p>


Aspect-based sentiment analysis (ABSA) extracts fine-grained **sentiment tuples** (e.g., ⟨aspect, opinion, sentiment⟩) from text. Prior template-based approaches relied on **static, fixed-order templates**, which often miss inter-element dependencies. Multi-view prompting remedies this by predicting tuples with multiple templates and aggregating them, but it introduces **heavy inference costs and out-of-distribution (OOD) errors**.

We propose **Dynamic Order Template (DOT)**, which **predicts—per instance—the minimal set of view orders actually needed**, ensuring both diversity and relevance. DOT achieves **higher F1 scores on ASQP and ACOS benchmarks** while **cutting inference time significantly**.

---

## 🧭 Table of Contents

* [Requirements & Environment](#requirements--environment)
* [Installation](#installation)
* [Quick Start](#quick-start)

  * [Train & Inference (Main Pipeline)](#train--inference-main-pipeline)
  * [Fine-tune LLM](#fine-tune-llm)
  * [Inference with Fine-tuned LLM](#inference-with-fine-tuned-llm)
* [Citation](#citation)
* [Contact](#contact)
* [License](#license)

---

## Requirements & Environment

* **Python**: 3.8
* **CUDA**: 11.6 (for the pinned Torch versions below)
* **PyTorch**: 1.12.1

Create and activate the conda environment:

```bash
conda create -n dot python=3.8 -y
conda activate dot
```

---

## Installation

Install PyTorch (CUDA 11.6 build) and the remaining dependencies:

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
```

---

## Quick Start

### Train & Inference (Main Pipeline)

```bash
bash scripts/run_main.sh
```

This script will **train the model per domain** and then **run inference**.

---

### Fine-tune LLM

Before running LLM scripts, **move into the `src` directory**:

```bash
cd src
python llms/sft_trainer.py
```

This fine-tunes the LLM and saves the checkpoint to the configured output path.

---

### Inference with Fine-tuned LLM

```bash
bash scripts/run_llms.sh

# (inside src/)
python llms/eval.py
```

Make sure you are still in the `src` directory when executing the Python scripts above.

---

## Citation

If you find this repo helpful, please cite our work:

```bibtex
@inproceedings{jun-lee-2025-dynamic,
    title = "Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis",
    author = "Jun, Yonghyun  and
      Lee, Hwanhee",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-short.48/",
    pages = "614--626",
    ISBN = "979-8-89176-252-7",
    abstract = "Aspect-based sentiment analysis (ABSA) assesses sentiments towards specific aspects within texts, resulting in detailed sentiment tuples.Previous ABSA models often used static templates to predict all the elements in the tuples, and these models often failed to accurately capture dependencies between elements. Multi-view prompting method improves the performance of ABSA by predicting tuples with various templates and then assembling the results. However, this method suffers from inefficiencies and out-of-distribution errors. In this paper, we propose a Dynamic Order Template (DOT) method for ABSA, which dynamically creates an order template that contains only the necessary views for each instance. Ensuring the diverse and relevant view generation, our proposed method improves F1 scores on ASQP and ACOS datasets while significantly reducing inference time."
}
```

> Replace author list and details with the final ACL entry once available.

---

## Contact

For questions or issues, please open a GitHub issue or contact: **\[zgold5670@cau.ac.kr]**

---

**Happy researching! 🚀**

