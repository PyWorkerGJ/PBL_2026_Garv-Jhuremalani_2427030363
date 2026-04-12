# Plant Disease Detector with AI-Generated Image Validation System

**PBL-2 | Manipal University Jaipur | CSE | 2025-2026**
**Student:** Garv Jhuremalani | **Reg No:** 2427030363
**Guide:** Dr. Ashok Kumar Saini

---

## What This Project Does

A two-stage deep learning system that:

1. **Checks if the leaf image is real or AI-generated** — ResNet-18 binary classifier (91.21% accuracy)
2. **Detects the tomato leaf disease** — ResNet-18 multi-class classifier (99.45% accuracy), only if image is authentic

If a fake/AI-generated image is uploaded, the pipeline stops immediately. No existing plant disease system does this.

---

## Live Presentation

[https://pyworkergj.github.io/PBL_2026_Garv-Jhuremalani_2427030363/](https://pyworkergj.github.io/PBL_2026_Garv-Jhuremalani_2427030363/)

---

## Project Structure

```
PlantDiseaseApp/
├── app.py
├── retrain.py
├── generate_fake_images.py
├── models/
│   ├── auth_model.pth
│   ├── best_disease_model.pth
│   └── class_names.json
├── disease/
│   ├── Tomato_Early_blight/
│   ├── Tomato_Late_blight/
│   ├── Tomato_Leaf_Mold/
│   └── Tomato_healthy/
├── training_data/
│   ├── real/
│   └── fake/
└── index.html
```

---

## Setup

```bash
pip install torch torchvision streamlit opencv-python Pillow
streamlit run app.py
```

---

## Training

```bash
python generate_fake_images.py
python retrain.py
```

---

## Model Architecture

Both models use ResNet-18 with pretrained ImageNet weights (Transfer Learning):

| Model | Type | Val Accuracy |
|---|---|---|
| Fake Detector | Binary (Real/Fake) | 91.21% |
| Disease Classifier | 4-class tomato diseases | 99.45% |

Final layer: `nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n))`
Optimizer: Adam (lr=0.0001, weight_decay=1e-4)
Training: 80/20 split, early stopping patience=4

---

## Disease Classes

- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_healthy

---

## Features

- Two-stage authenticity + disease pipeline
- Transfer learning (pretrained ResNet-18)
- Webcam capture via st.camera_input()
- Treatment recommendations per disease
- Confidence threshold warning (< 60%)
- Session prediction history log
- Early stopping during training

---

## Results

```
Fake Detector Val Accuracy:       91.21%
Disease Classifier Val Accuracy:  99.45%
Disease training images:          5,452 (4 classes)
Fake detector training images:    2,560 (balanced)
```

---

## Dataset

Disease images: [PlantVillage Dataset](https://plantvillage.psu.edu)
Fake images: Custom generated using generate_fake_images.py

Model .pth files not included due to size — run training pipeline locally.

---

## Key References

1. Mohanty et al. (2016) — Frontiers in Plant Science
2. Karthik et al. (2020) — Applied Soft Computing
3. Hasan et al. (2023) — Plants journal
4. Shoaib et al. (2025) — Frontiers in Plant Science
5. Ashurov et al. (2025) — Frontiers in Plant Science
6. Wang et al. (2025) — Journal of King Saud University

---

Department of Computer Science and Engineering, Manipal University Jaipur
