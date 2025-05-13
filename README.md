# Deep Learning SP25: Adversarial Robustness Evaluation (Project 03)

**Team:**  
- Rujuta Amit Joshi ([rj2719@nyu.edu](mailto:rj2719@nyu.edu))  
- Lavanya Deole ([lnd2037@nyu.edu](mailto:lnd2037@nyu.edu))  
- Sarang Kadakia ([sk11634@nyu.edu](mailto:sk@nyu.edu))  

**Course:** Deep Learning (Spring 2025)  
**Institution:** NYU Tandon School of Engineering  

---

## Overview

This project explores the vulnerability of deep convolutional neural networks (CNNs) to adversarial examples. We implemented and evaluated three attack methods on a pre-trained ResNet-34 model using a 500-image ImageNet subset:

- **FGSM (Fast Gradient Sign Method)**
- **PGD (Projected Gradient Descent)**
- **Iterative Patch Attack**

We further evaluated the **transferability** of these adversarial samples to DenseNet-121, visualized perturbations, and saved 500 image samples per attack for inspection.

---

## Attacks & Parameters

| Attack | ε | Steps | Description |
|--------|----|--------|-------------|
| FGSM   | 0.02 | 1 | Fast one-step gradient method |
| PGD    | 0.02 | 15 | Iterative FGSM with ε-ball projection |
| Patch  | 0.3 / 0.5 | 50 | Localized patch (32×32), repositioned every 10 steps |

---

## Results

### Accuracy on ResNet-34

| Attack   | Top-1 (%) | Top-5 (%) |
|----------|-----------|-----------|
| Original    | 76.00     | 94.20     |
| FGSM     | 3.40      | 21.20     |
| PGD      | 0.00      | 1.40      |
| Targeted Patch ε=0.5 | 0.00  | 13.00     |
| Untargeted Patch ε=0.5 | 0.00  | 13.00     |

### Transferability to DenseNet-121

| Attack   | Top-1 (%) | Top-5 (%) |
|----------|-----------|-----------|
| Original    | 76.00     | 94.20     |
| FGSM     | 45.60     | 76.20     |
| PGD      | 35.60     | 72.40     |
| Targeted Patch ε=0.5 | 58.80  | 87.60     |
| Untargeted Patch ε=0.5 | 57.40  | 85.40     |

---

## Visualizations

For each attack, we provide:
- Clean vs. adversarial images (5 samples)
- Difference heatmaps (×5 amplified for visibility)

FGSM (ε=0.02) visualization of 3–5 adversarial examples

![FGSM_ε=0 02](https://github.com/user-attachments/assets/b1915c3f-2116-4fa7-ad6a-ab850ac1373c)

PGD (ε=0.02) visual comparison

![PGD_ε=0 02](https://github.com/user-attachments/assets/1dd58a1f-47ff-43a7-800f-0f92302baa76)

Patch Attack (ε=0.3) visualization

![Patch_Attack_ε=0 3](https://github.com/user-attachments/assets/531acf96-4310-4526-96f3-816a8ba9a410)

Patch Attack (ε=0.5) visualization

![Patch_Attack_ε=0 5](https://github.com/user-attachments/assets/b5096589-2b56-4a75-8da5-c06a330a882b)


**Saved image folders**:

```
AdversarialTestSet1/ → FGSM (500 samples)
AdversarialTestSet2/ → PGD (500 samples)
AdversarialTestSet3/ → Patch ε=0.5 (500 samples)
```


# Plots included:

- `final_accuracy_barplot`

![final_accuracy_barplot](https://github.com/user-attachments/assets/2df59ec9-64bc-44d3-a732-ed0d021e0c4b)


- `final_top5_accuracy_barplot`

![final_top5_accuracy_barplot](https://github.com/user-attachments/assets/433a4872-8acf-4f4c-92f8-b753149d3ea4)


- `patch_accuracy_vs_epsilon`

![patch_accuracy_vs_epsilon](https://github.com/user-attachments/assets/cd3e6899-e1e6-4ab9-b989-39b534160dd6)


---

## Lessons Learned & Mitigation

- PGD and Patch attacks cause severe misclassification; FGSM is fast but less effective.
- Transferability observed in FGSM and PGD across ResNet and DenseNet.
- Visualization proved key for understanding imperceptibility vs. model failure.

**Mitigation Strategies**:
- Adversarial training
- Input preprocessing (e.g., JPEG compression)
- Model ensembling
- Gradient regularization

---

## Environment & Setup

```bash
pip install torch torchvision matplotlib tqdm
```

Run all experiments in project3_colab_notebook.ipynb

Compatible with Google Colab (GPU runtime recommended)

No additional training was performed; all models are pre-trained (ImageNet)


# Directory Structure
```
DLProject03/
│
├── DLProj03.ipynb                # Main notebook
│
├── Visualizations/               # All attack visualizations
│   ├── FGSM_ε=0.02.png
│   ├── PGD_ε=0.02.png
│   ├── Patch_Attack_ε=0.3.png
│   ├── Patch_Attack_ε=0.5.png
│   ├── final_accuracy_barplot.png
│   └── patch_accuracy_vs_epsilon.png
│
├── rj2719_lnd2037_sk11634_DLProjectReport03.pdf
├── AdversarialTestSet1.zip       # FGSM ε=0.02
├── AdversarialTestSet2.zip       # PGD ε=0.02
├── AdversarialTestSet3.zip       # Patch ε=0.5
│
└── README.md                     # Full project overview
```
