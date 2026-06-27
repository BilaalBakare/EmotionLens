# EmotionLens

Real-time facial emotion recognition, built from scratch with deep learning.

EmotionLens detects and classifies 7 human emotions — **angry, disgust, fear, happy, neutral, sad, surprise** — directly from a live webcam feed, using a fine-tuned EfficientNet model and OpenCV for face detection.

[Live Demo](#) · [Demo Video](#) · [LinkedIn](https://www.linkedin.com/in/bilaal-bakare) · [GitHub](https://github.com/BilaalBakare)

---

## Why This Project Exists

Most AI systems are emotionally blind — they can detect that a face is present, but have no sense of how that person feels. That gap matters in places where emotional state changes the right course of action:

- **Driver safety** — detecting drowsiness or distress before it becomes dangerous
- **Mental health support** — giving therapists and care teams an objective, continuous signal
- **Adaptive learning** — letting e-learning platforms notice confusion or disengagement in real time
- **Human-computer interaction** — making interfaces that respond to how a person actually feels, not just what they click

EmotionLens is a hands-on exploration of that problem: building an emotion-aware system end to end, from raw pixels to a live, working demo.

---

## How It Works

```
Webcam frame
   ↓
Face detection (Haar Cascade)
   ↓
Crop + CLAHE lighting correction
   ↓
Resize to 224×224, normalize
   ↓
Fine-tuned EfficientNet (PyTorch)
   ↓
Temperature-scaled softmax + multi-frame smoothing
   ↓
Emotion label + confidence, displayed live
```

**Stack:** Python · PyTorch · OpenCV · EfficientNet (transfer learning) · Gradio

---

## The Build Journey

This project went through several real iterations, each one surfacing a genuine problem and a genuine fix:

| Stage | What Happened |
|---|---|
| **Baseline (FER-2013)** | First working model, fine-tuned EfficientNet-B0. Worked, but showed poor accuracy, frozen-at-1.00 overconfidence, and predictions that flickered between classes frame to frame. |
| **Calibration fixes** | Added **temperature scaling** to soften overconfident softmax outputs, and a **7-frame prediction buffer** that averages probabilities over time instead of trusting any single frame. |
| **Dataset upgrade** | Switched to **AffectNet** (Kaggle subset) and a heavier **EfficientNet-B4**, unfreezing deeper layers for better feature adaptation. |
| **Class imbalance bug** | The model defaulted to predicting "sad" far too often. Root cause: sad had ~3,000+ images vs. disgust's ~1,200. Fixed with **weighted cross-entropy loss** (inverse-frequency weighting) and **label smoothing**. |
| **Lighting robustness** | Added **CLAHE** (Contrast Limited Adaptive Histogram Equalization) preprocessing to normalize lighting before inference. |
| **Data augmentation experiment** | Tried aggressive brightness/contrast/blur augmentation to improve real-world robustness. It underperformed the non-augmented version on this run — likely too aggressive for the epoch budget — so the moderate version was kept as the shipped model. |
| **Model checkpointing** | Training now saves the best-performing epoch by validation accuracy, not just the last one. |
| **Live deployment** | Wrapped the model in a **Gradio** app for an in-browser demo, deployed on Hugging Face Spaces. |

---

## Current Results

- **~68% test accuracy** on the AffectNet subset, after fixing class imbalance
- **7 emotion classes**, trained with weighted loss to reduce majority-class bias
- For context: human inter-rater agreement on FER-2013 is only **~65%** — even people disagree on these labels that often, so this is close to the practical ceiling for this kind of dataset, not a tuning failure

### A known, honest limitation

The live web demo (Gradio, browser-based webcam capture) is measurably **less accurate and less confident** than the local OpenCV version (`webcam.py`). This was investigated in depth — ruled out channel order (RGB vs BGR), resolution mismatches, and false face detections one by one with direct evidence. The remaining gap appears to be **train-serve skew**: the browser's camera pipeline (auto-exposure, internal compression, JavaScript-mediated capture) produces images with subtly different statistical properties than OpenCV's direct camera access, which is what the model was effectively validated against locally. This is a well-documented category of real-world ML deployment problem, not a bug in a specific line of code — the long-term fix is training with stronger, more targeted augmentation to close that gap, which is the current top priority below.

---

## Project Structure

```
EmotionLens/
├── data/                  # FER-2013 / AffectNet (not committed — see .gitignore)
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
├── src/
│   ├── model.py           # Emolens — EfficientNet architecture + classifier head
│   ├── train.py           # Training + evaluation loop, checkpointing
│   └── webcam.py          # Local real-time inference (OpenCV direct capture)
├── models/                # Saved .pth weights (not committed)
├── app.py                 # Gradio web app for live deployment
└── README.md
```

---

## Running It Locally

```bash
git clone https://github.com/BilaalBakare/EmotionLens.git
cd EmotionLens
pip install -r requirements.txt
python src/webcam.py
```

For best results, ensure your face is clearly visible and well lit.

---

## What's Next

- **Larger, richer data** — pursuing full AffectNet access and blending in RAF-DB for genuine real-world capture diversity
- **Robustness training** — augmentation specifically targeted at closing the browser-vs-local capture gap, rather than generic augmentation
- **Deeper evaluation** — per-class confusion analysis to target exactly which emotions are still being confused
- **Tighter deployment** — narrowing the accuracy gap between the local demo and the live Hugging Face version

---

## License

Code is released under the MIT License. The trained model weights were fine-tuned on FER-2013 and AffectNet, both of which carry non-commercial / research-use restrictions — the weights are shared here for educational and portfolio purposes under those same terms.

---

## Acknowledgements

Built by **Bilaal Bakare** as a hands-on deep learning project — going from a from-scratch neural network (MNIST, 94.5% accuracy) to a deployed, real-time computer vision system.
