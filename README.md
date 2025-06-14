# 🚀 Naïve Bayes - Credit Risk Prediction  

## 📌 Overview  
Implementation of a Naïve Bayes classifier to predict credit risk, achieving **84.5% accuracy** (Gaussian distribution).

## Dependencies
- pandas
- sklearn
- pickle
- yellowbrick

## ⚙️ Hyperparameters  
python
GaussianNB(
    priors=None,           # Class priors (automatically adjusted)
    var_smoothing=1e-9     # Default stability for zero-variance features
)

No hyperparameter tuning was needed for this use case

## Preprocessing
**Categorical Encoding**

-Challenge: Naïve Bayes assumes numerical inputs.
-Methods Tested:

|     Encoding      |  Accuracy  |
|-------------------|------------|
|      One-Hot      |   ~80.0%   |
|  Target Encoding  |   ~84.5%   |


**Feature Correlation Insight**

High correlation between:
- `person_home_ownership`
- `loan_intent`
- `loan_grade`

Solution: Target encoding preserved relationships while reducing dimensionality vs. one-hot.

**Key Takeaways**

Gaussian outperformed Bernoulli, suggesting near-normal data distributions.

## Performance Metrics

![Confusion Matrix](images/naive_bayes_cm.png)
![Classification Report](images/naive_bayes_cf.png)
