# 📏 Logistic Regression - Credit Risk Prediction  

A simple one layer weight based model used to achieve 86.8% accuracy

## 📌 Overview  
Implementation of a **Logistic Regression** classifier to predict high-risk loans. Key wins:

    86.8% accuracy (best among tested configurations).

    Data Scailing ensured a 4.8% accuracy increase

Business Impact: This model could help lenders reduce high-risk loan approvals, potentially saving millions in defaults.

## Dependencies
- pandas
- sklearn
- pickle
- yellowbrick

## ⚙️ Hyperparameters  
python
GaussianNB(
    random_state = 1    #
)

## Preprocessing
**Data Cleaning**

    Inconsistent Data:Dropped illogical records(e.g. person_age / person_emp_length > 100)

    Missing Data:
    
        person_emp_length: Nulls were imputed with 0 (assuming nulls indicated no employment history).

        loan_int_rate: Nulls were replaced with the global mean, as these likely represented lost data.


**Scailing**

While testing preprocessing methods with the purpose of training the model it was clear that the technique that showed the biggest positive impact in the algorithm predictive power was the scailing as the table below indicate

|  Scailing  |  Accuracy  |
|------------|------------|
|     No     |   ~82.0%   |
|  Standard  |   ~86.8%   |

**Categorical Variable Encoding Method**

Contrary to my expectations the algorithm was very effective dealing with high number of dimensions as the algorythm was effective when trained with one hot encoded data

|  Encoding  |  Accuracy  |
|------------|------------|
|  One-Hot   |   ~86.7%   |
|  Target    |   ~86.8%   |



## Key Takeaways

**Effective with high dimensions**: The algorythm didnt show signs of poor perfomance or low time training efficiency when submitted to a one-hot encoded database

**Data scale difference**: Data scailing was the key technique to ensure the algorythm good performance

## Performance Metrics

![Confusion Matrix](images/knn_cm.png)

![Classification Report](images/knn_cf.png)
