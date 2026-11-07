# Machine-Learning-Assignment-2
Model analysis on kaggle dataset

a. **Problem Statement** - Comparative model analysis for a bank customer churn dataset (Binary Classification).
  
b. **Dataset Description** - The dataset contains associated bank customer details to determine whether the customer will churn or not.

c. **GitHub Repository Link** - https://github.com/hafffsah/Machine-Learning-Assignment-2

d. **Models Used**

    - Comparison Table with evaluation metrics calculated for all 5 models:  
    
      |   ML Model Name        |  Accuracy  |    AUC    |  Precision  |  Recall  |    F1    |  MCC    |
      |:-----------------------|:----------:|:---------:|:-----------:|:--------:|:--------:|--------:|
      |   Logistic Regression  |  .7970     |  .5649    |  .5263      |  .1695   |  .2564   |  .2109  |
      |   Decision Tree        |  .7905     |  .6802    |  .4630      |  .5000   |  .4808   |  .3502  |
      |   KNN                  |  .8205     |  .6466    |  .6404      |  .3452   |  .4485   |  .3767  |
      |   Naive Bayes          |  .8270     |  .6621    |  .5677      |  .3953   |  .4660   |  .3753  |
      |   SVM                  |  .8575     |  .6851    |  .8351      |  .3904   |  .5320   |  .5072  |
 
    
    - OBSERVATIONS
    
      |  ML Model Name        |  Observation about model performance                                                          |
      |:----------------------|----------------------------------------------------------------------------------------------:|
      |  Logistic Regression  |  Poor recall rate, as well as barely there metrics. Dataset imbalance contributes to this,    | 
      |                       |  has the lowest TPs observed owing to lesser +ves in dataset. AUC nearly .5, random model.    |  
      |  Decision Tree        |  Highest recall, confusion matrix points to similar values in comparison to other models due  |
      |                       |  to nature of dataset, lowest precision, likely to cull churned in comparison.                |
      |  KNN                  |  Average performance, slightly on the better side.                                            |  
      |  Naive Bayes          |  Decent accuracy, remaining metrics similar to other classifiers. Model can be improved by    | 
      |                       |  leveraging specific classifier type depending on the feature types.                          |
      |  SVM                  |  Highest value on all evaluation metrics except recall (has a lower TP rate), model performs  |
      |                       |  better due to better generalization.                                                         |                      
