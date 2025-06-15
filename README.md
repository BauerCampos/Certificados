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
**Inconsistent data**

Since this database had very little amount of data with clear signs of inconsistent data(the variables person_age and person_emp_length across a few lines) i decided to simply delete those lines as most of them showed signes of inconsistent values on both of the variables

**Missing data**

This database had missing data on only two variables, those being person_emp_legth and loan_int_rate, and each variable had it's missing value treated differently given the fact that both had different natures

    person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

    loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.


**Encoding**
The database used for this project have 3 categorical variables(person_home_ownership, loan_intent and loan_grade), since the Naïve Bayes doesn't handle categorical values the database needed encoding on said variables, on further analysis those variables were classified as nominal variables, this meant there was no value order on any of them which discarded **Label Encoding** as a possibility, further analysis on each variable individually showed that all of them had great isoled impact on the target variable value and for this reason target encoding was the most sucessful option between the two tested methods as showed below

|     Encoding      |  Accuracy  |
|-------------------|------------|
|      One-Hot      |   ~80.0%   |
|  Target Encoding  |   ~84.5%   |

**Scailing**

The theory behind the algorithm Naïve bayes suggest that the scailing of the data is not necessary, but in this case the massive scale difference between variables drastically altered the algorithm performance suggesting there is reason to test the necessity of data scailing when using this algorithm

|     Scailing      |  Accuracy  |
|-------------------|------------|
|        Not        |   ~81.0%   |
| Standard Scailing |   ~84.5%   |

**Key Takeaways**

Gaussian outperformed Bernoulli, suggesting near-normal data distributions.

## Performance Metrics

![Confusion Matrix](images/naive_bayes_cm.png)

![Classification Report](images/naive_bayes_cf.png)
