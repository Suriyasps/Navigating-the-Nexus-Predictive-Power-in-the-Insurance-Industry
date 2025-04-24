# Navigating the Nexus: Predictive Power in the Insurance Industry

# Data Mining  
**Porject By**: Suriya Subbiah Perumal  

---

## 📌 Project Overview

This project supports Imperial Ltd.'s life insurance marketing initiative by building a **predictive model** using customer demographic, behavioral, and transactional data. The goal: **identify potential buyers of life insurance** to enable **targeted marketing** and increase ROI.

---

## 🧭 Methodology: CRISP-DM Framework

1. **Business Understanding**: Use predictive analytics to identify high-likelihood buyers.
2. **Data Understanding**: Explore demographic and financial indicators across 4,000+ records.
3. **Data Preparation**:
   - Cleaned missing and inconsistent values
   - Used NLP-inspired logic to preprocess categorical fields (e.g., standardizing 'marriage', 'child', 'online' fields)
4. **Modeling**:
   - Random Forest
   - Support Vector Machine (SVM)
   - Artificial Neural Networks (ANN)
5. **Evaluation**: Precision, recall, F1 score, ROC-AUC
6. **Deployment**: Prepare model insights for use in marketing targeting tools

---

## 🧮 Models Used

### 🌲 Random Forest
- Baseline accuracy: **88.4%**, improved to **89.6%** via hyperparameter tuning.
- Feature importance: `'online'` is the most predictive variable.
- ROC-AUC: **0.93**, showing strong class separation.
- Hyperparameters tuned via `GridSearchCV`.

### 💡 Support Vector Machine (SVM)
- Multiple models with polynomial and linear kernels.
- Final tuned model: **89% accuracy**, AUC of **0.91**.
- Conservative prediction threshold to avoid false positives.

### 🧠 Artificial Neural Network (ANN)
- Input layer: 128 neurons, ReLU activation
- Dropout layers used to prevent overfitting
- Accuracy: **~90%**
- ROC-AUC: **0.93**
- F1 score optimized at **threshold 0.81**

---

## 📊 Visual Analysis Highlights

- **Gender & Life Insurance**: Males more likely to purchase
- **Marital Status**: Married individuals more inclined toward life insurance
- **Family Income & Region**: Strong geographic and economic segmentation
- **Occupation**: Professionals are top buyers of life insurance
- **Correlation Matrix**: Digital engagement and age most impactful

---

## 🧠 Tools & Technologies

- **Language & Libraries**: Python (`pandas`, `seaborn`, `scikit-learn`, `keras`, `matplotlib`)
- **NLP Techniques**:
  - Standardization of demographic attributes based on context (e.g., child, marriage)
  - Categorical encoding informed by text processing strategies
- **ML Algorithms**:
  - `RandomForestClassifier`
  - `SVC`
  - `Keras Sequential API`

---

## 📈 Evaluation Metrics

| Model      | Accuracy | Precision | Recall | F1 Score | AUC   |
|------------|----------|-----------|--------|----------|-------|
| Random Forest | 89.6%    | High      | High   | High     | 0.93  |
| SVM        | 89%      | High      | High   | High     | 0.91  |
| ANN        | 90%      | Balanced  | Balanced| Optimal at 0.81 | 0.93  |

---

## ✅ Conclusion

- The **ANN model** is recommended due to its superior ROC performance and balanced F1 score.
- Predictive modeling proves highly effective in **targeting potential customers**, helping Imperial Ltd. optimize its marketing and resource allocation.
- Future applications can integrate real-time customer data and include LLM-based sentiment analysis from customer feedback.

---

## 📚 References

Key references include:
- Breiman (2001) on Random Forests
- Cortes & Vapnik (1995) for SVM theory
- Srivastava et al. (2014) on Dropout in Neural Nets
- Chapman et al. (2000) on CRISP-DM methodology
- Derrig et al. (2002), Leibenberg et al. (2012) for insurance domain insights

Full citations are included in the appendix of the main report.

---

## 💬 Future Work

- Integrate LLMs for text-based feature enrichment (e.g., analyzing support logs or survey text).
- Deploy model via API in marketing platforms.
- Use SHAP for model interpretability in business dashboards.

