# Dogs vs. Cats Classification with EfficientNet

This project tackles the classic image classification task from the [Kaggle "Dogs vs. Cats Redux" competition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition), where the goal is to classify images as either **dog** or **cat**. The final model was submitted to Kaggle and evaluated based on classification error.

---

## 📁 Dataset Overview

- **Total Images**: 25,000 (12,500 dogs + 12,500 cats)
- **Input Format**: RGB JPEG images (~250x250 px)
- **Labels**: Binary (0 = cat, 1 = dog)

---

## 🧠 Modeling Approach

### ✅ Transfer Learning with EfficientNetB0

We leveraged **EfficientNetB0**, a pretrained convolutional neural network trained on ImageNet. The training followed a **two-phase transfer learning strategy**:

#### Phase 1: Feature Extraction
- **Logic**: Initially froze all pretrained layers to retain generalized feature detectors (edges, shapes, textures).
- **Only the final classification head was trained**.
- **Why?**: Prevents early-stage overfitting and allows the final layers to learn dog-vs-cat-specific patterns while leveraging generic vision knowledge.

#### Phase 2: Fine-Tuning
- **Unfroze a subset of the final layers of the EfficientNet base model**
- Trained the entire model (with a smaller learning rate)
- **Why?**: Fine-tuning allows the deeper convolutional layers to adapt to dataset-specific features and boost performance beyond just using generic features.

---

### 🔁 Cross-Validation for Robustness

To ensure the model's performance generalized well across different data splits, we implemented **5-fold cross-validation** during training. This helped evaluate variance in performance, reduce overfitting, and confirm the model’s stability across diverse subsets of the training data.

---

## 🧪 Model Evaluation & Results

- **Validation Accuracy (EfficientNet only)**: ~94%
- **Private Leaderboard Score (EfficientNet)**: **0.05789**
- **Comparison Model**: Ensemble approach (score: 0.14257)
- **Conclusion**: EfficientNet with two-stage transfer learning and cross-validation significantly outperformed naive or ensemble-based models in generalization and classification accuracy.

---

## 🖼️ Kaggle Submission Screenshot

![Kaggle Submission Screenshot](Images/Kaggle_Submission.png)
