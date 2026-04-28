# Applied Diffusion Models: Exploration, Control, and Adaptation

This repository contains the implementation and analysis for **Assignment 2: Applied Diffusion Models** of the Advanced Deep Learning course (Spring 2026). The project explores the theoretical and practical aspects of Latent Diffusion Models (LDMs) using the Hugging Face `diffusers` library.

## 🚀 Project Overview

The project is divided into three core parts:
1.  **Pipeline Exploration & Scheduler Analysis:** Systematic study of numerical solvers, sampling steps, and Classifier-Free Guidance (CFG).
2.  **Controlled Generation (SDEdit):** Implementing image-to-image translation (Day-to-Night) and analyzing the fidelity-creativity tradeoff.
3.  **Domain Application (LoRA):** Capturing specialized artistic styles (Ukiyo-e) using Parameter-Efficient Fine-Tuning.

## 📂 Project Structure

```text
assignment2/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── part1_exploration.py    # Task 1: Schedulers, Steps, and CFG Analysis
├── part2_controlled.py     # Task 2: SDEdit (Img2Img) implementation
├── part3_domain.py         # Task 3: LoRA stylization and ablation study
├── prepare_part3_data.py   # Data preparation for style learning
├── report.tex              # Comprehensive academic report (NeurIPS template)
├── checklist.tex           # NeurIPS reproducibility checklist
├── utils/
│   └── metrics.py          # CLIP Score and LPIPS evaluation utilities
├── data/                   # Input datasets (generated)
└── outputs/                # Experimental results and visual grids
```

## 🛠️ Installation & Setup

### Environment
The project was developed using **Python 3.10** and **CUDA 13.1** on an **NVIDIA RTX 5060 Ti**.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd assignment2
    ```

2.  **Create and activate Conda environment:**
    ```bash
    conda create -n diffusion_env python=3.10 -y
    conda activate diffusion_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🧪 Experiments & Tasks

### Part 1: Pipeline Exploration
- **Schedulers Compared:** DDPM, DDIM, Euler Discrete, and DPM-Solver Multistep.
- **Analysis:** Impact of inference steps (5–100) and CFG scale (1.0–20.0) on semantic alignment.
- **Key Metric:** CLIP Score for text-image alignment.

### Part 2: Controlled Generation (Track B - SDEdit)
- **Task:** "Day-to-Night" landscape translation.
- **Analysis:** Sweeping the `strength` parameter to find the "sweet spot" between preserving source structure and adhering to target semantics.
- **Metrics:** CLIP Score vs. LPIPS (Learned Perceptual Image Patch Similarity).

### Part 3: Domain Application (Track A - LoRA)
- **Style:** Ukiyo-e (Japanese Woodblock Print).
- **Technique:** PEFT using LoRA (Low-Rank Adaptation).
- **Analysis:** Ablation study on LoRA Rank ($r=4, 16, 64$) and its impact on style generalization to modern subjects.

## 📊 Key Results

- **Optimal CFG:** 7.5 to 10.0 provides the best balance; values >15.0 lead to saturation artifacts.
- **DPM-Solver Efficiency:** Achieves high structural stability in as few as 10–20 steps.
- **SDEdit Sweet Spot:** A strength of **0.6** effectively translates the domain while preserving the original landscape's composition.
- **LoRA Rank:** Rank **16** captures stylistic essence without overfitting to the small reference set.

## 📝 Report
A detailed 10+ page report following the **NeurIPS 2026 template** is available in `report.tex`, providing theoretical derivations (Probability Flow ODEs, CFG extrapolation) and in-depth qualitative analysis.

## 🎓 Acknowledgments
- Course: Advanced Deep Learning (Spring 2026)
- Base Model: [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- Library: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
