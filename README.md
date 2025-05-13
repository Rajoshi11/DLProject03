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

| Attack                  | Top-1 (%) | Top-5 (%) |
|-------------------------|-----------|-----------|
| Clean (No attack)       | 76.00     | 94.20     |
| FGSM (ε = 0.02)         | 3.80      | 21.00     |
| PGD  (ε = 0.02)         | 0.00      | 1.40      |
| Targeted Patch (ε = 0.3)| 3.80      | 25.60     |
| Targeted Patch (ε = 0.5)| 3.00      | 17.60     |
| Untargeted Patch (ε = 0.3) | 3.60   | 38.20     |
| Untargeted Patch (ε = 0.5) | 2.00   | 26.00     |

### Transferability to DenseNet-121

| Attack                   | Top-1 (%) | Top-5 (%) |
|--------------------------|-----------|-----------|
| Clean (No attack)        | 74.60     | 93.60     |
| FGSM → DenseNet-121      | 45.60     | 75.80     |
| PGD → DenseNet-121       | 34.80     | 73.20     |
| Targeted Patch → DenseNet-121 (ε = 0.5) | 69.20 | 89.20 |
| Untargeted Patch → DenseNet-121 (ε = 0.5) | 64.80 | 89.20 |

---

## Visualizations

For each attack, we provide:
- Clean vs. adversarial images (5 samples)
- Difference heatmaps (×5 amplified for visibility)

FGSM (ε=0.02) visualization of 3–5 adversarial examples

![FGSM_ε=0 02](https://github.com/user-attachments/assets/9c313347-1c41-4f67-8b55-608261f67d28)

PGD (ε=0.02) visual comparison

![PGD_ε=0 02](https://github.com/user-attachments/assets/8b1997f7-0735-4be5-9947-7fc360234749)

Targeted Patch Attack (ε=0.3) visualization

![Targeted_Patch_Attack_ε=0 3](https://github.com/user-attachments/assets/a8812999-95dc-4294-a233-28f7aa08bb7a)

Targeted Patch Attack (ε=0.5) visualization

![Targeted_Patch_Attack_ε=0 5](https://github.com/user-attachments/assets/535d1c8a-9fe6-4438-b8d9-17e74fa62cb6)

Untargeted Patch Attack (ε=0.3) visualization

![Patch_Attack_ε=0 3](https://github.com/user-attachments/assets/2840b7e3-4801-40fe-80cd-e18c883c7635)

Untargeted Patch Attack (ε=0.5) visualization

![Patch_Attack_ε=0 5](https://github.com/user-attachments/assets/df52fe2f-c30d-42cb-b351-de8671897533)


**Saved image folders**:

```
AdversarialTestSet1/ → FGSM (500 samples)
AdversarialTestSet2/ → PGD (500 samples)
AdversarialTestSet3/targeted → Patch ε=0.5 (500 samples)
AdversarialTestSet3/untargeted → Patch ε=0.5 (500 samples)
```


# Plots included:

- `final_top1_accuracy_barplot`

![final_accuracy_barplot](https://github.com/user-attachments/assets/a11fab1f-fdc6-4be1-bf9c-59514ac349b6)


- `final_top5_accuracy_barplot`

![final_top5_accuracy_barplot](https://github.com/user-attachments/assets/321d3eed-0961-4056-8fcc-8c5021857eae)


- `patch_accuracy_vs_epsilon`

![patch_accuracy_vs_epsilon](https://github.com/user-attachments/assets/c5f27e63-c999-4ff0-b14e-881ddf333d52)


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
│   ├── Targeted_Patch_Attack_ε=0.3.png
│   ├── Targeted_Patch_Attack_ε=0.3.png
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
