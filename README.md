# Attrition prediction

In case of this issue, attrition prediction is formulated as a binary classification type of task. Throughout the task, I've followed the standard data science process:
1. Explorative data analysis
2. Data transformation + feature selection
3. Pattern search
4. Evaluation

*Note: data science process is highly iterative, therefore the insights gained at each step were used to improve the previous steps. Also in this case data gathering and raw data preparation wasn't necessary in a classical sense, because the dataset exhibited relatively good quality (e.g. well-defined features, no missing values)*

## Repository structure

- The repository was set up with a data folder to contain the source data, which is unconventional but doable due to its limited size
- There is also a reports folder to contain the result of the EDA
- The task description was put into a different folder as well for having a cleaner understanding of the repo structure
- *final_results.ipynb* contains the chosen models, while *sandbox.ipynb* was the lab environment
- Finally, utils contain useful custom scripts that helped the iterative process
- The process is detailed in this README description, the experimentation can be found in *main.ipynb*

## Baseline / Goal

The point of setting up a baseline is to decrease time-to-market, and to have an initial idea what to improve upon. The baseline chosen was a default *KNeighborsClassifier* with looking at the most similar person. It was a straightforward first choice because of its explainability and simplicity. The data was filtered down to the numerical values to be able to get baseline results quickly. This baseline helped me determine the areas to be improved, and the metrics to use for evaluation. The baseline results were the following by using 10-fold stratified cross-validation:
- Accuracy: 75.99%
- Recall: 25.33%
- Precision: 25.04%
- F1-score: 24.96%

The first metric used was accuracy, which was suspiciously convincing for a baseline, therefore I checked if we have an unbalanced issue at hand by checking the recall and precision (and their harmonic mean, the F1-score) which confirmed the hypothesis, therefore accuracy is no more the primary metric of model goodness but more suitable ones for unbalanced issues such as recall, precision and F1-score.

## Explorative data analysis (EDA)

EDA can be a tedious task, therefore (for the simplest parts) I used an automated EDA tool, [*Sweetviz*](https://github.com/fbdesignpro/sweetviz "Sweetviz homepage") to gain insights into the data I'm dealing with. For this, a report was prepared (*report.html*) which helped drawing some conclusions / making a few decisions:
1. The following features were dropped due to having 1 distinct value, therefore having no predictive value: *EmployeeCount*, *Over18*, *StandardHours*
2. The following feature was dropped due to being a unique identifier, therefore having no predictive value: *EmployeeNumber*
3. There is no suspicion of corrupted data, because no outliers were spotted in the report
4. This classification issue can be qualified as an unbalanced issue, since there are significantly (appr. 6×) more negative samples than positive ones. This is further comfirmed by the baseline experiment, where a quite convincing accuracy (around 76%) can be obtained, yet the recall remains realtively poor (around 25%).
5. The computed correlation matrix (w.r.t the target value) showed the first promising features, e.g. *Age*, *MonthlyIncome*, *TotalWorkingYears*. These are also making sense by only using common sense, therefore the quality of the features were asserted through both common sense and the tools of mathematics.

At the end, the values are classified and treated as (some are classified as nominal/ordinal features by the report preparing library, but this is overwritten by the task description which allows me to treat them as numerical ones, which is the less strict option):
1. Target variable (binary variable): *Attrition*
2. Numerical features: *Age*, *DailyRate*, *DistanceFromHome*, *Education*, *EnvironmentSatisfaction*, *HourlyRate*, *JobInvolvement*, *JobLevel*, *JobSatisfaction*, *MontlyIncome*, *MonthlyRate*, *NumCompaniesWorked*, *PercentSalaryHike*, *PerformanceRating*, *RelationshipSatisfaction*, *StockOptionLevel*, *TotalWorkingYears*, *TrainingTimesLastYear*, *WorkLifeBalance*, *YearsAtCompany*, *YearsInCurrentRole*, *YearsSinceLastPromotion*, *YearsWithCurrManager*
3. Categorical features: *BusinessTravel*, *Department*, *EducationField*, *Gender*, *JobRole*, *MaritalStatus*, *OverTime*

## Data transformation + feature selection

### Data transformation / Feature engineering

Categoricals to numericals:
- Binary categorical features: transform to numerical features with a value of either 0 or 1 (this is easier for machine learning models to handle)
- Non-binary categorical features: one-hot encoding

Dimension reduction:
- PCA: apply to the whole dataset or highly correlated / similar features

Unbalance correction: random and synthetic over- and undersampling

Scaling:
- StandardScaler
- MaxAbsScaler
- MinMaxScaler

### Feature selection

The problem of feature selection is approached from a top-down direction: recursive feature elimination, this is to use an exhaustive, brute force search to see the effect of ignoring features in a recursive manner

## Pattern search (modelling)

The modelling approaches used here are broken down into three categories from explainability and simplicity point of view.

### Base models

1. K-Neighbors Classifier
2. Decision Tree
3. Support Vector Machine
4. Logistic Regression

### Ensemble models

1. Gradient Boosting (+XGBoost)
2. Random Forest
3. AdaBoost

### Deep learning

Due to the size and nature of the data, anything besides a small size MLP is ill-advised, because deep learning approaches are limited in explainability, are black box in nature, and generally require more data.

## Experiment log

1. Performed KNN on numericals to obtain a baseline (only used the closest neighbor)
2. Used one-hot encoding to include categorical features, this didn't improved (or changed) on KNN
3. Turned to Gradient Boosting, which was improved by the transformed categorical features
4. Added PCA into the workflow, which shows a slight improvement, but not significant (compared to the standard deviation) enough to actually deem it improvement, although it seems to have a metric stabilizing effect during cross-validation
5. Now turned to resampling, to correct the balance between the minority and majority classes, first one was RandomUnderSampler with equal ratios, this resulted in significantly better recall, but worse precision
6. Experiments were conducted with synthetic methods like SMOTE, ADASYN or SMOTEENN, they haven't provided an edge on random resampling
7. The next experiment was to test the effect of data scaling on the result, where MinMaxScaler was able to slightly fix the precision degradation coming from resampling with only a minor decrease in recall
8. Finally, to conclude the data preprocessing/transformation part, I've conducted RFE feature selection, but it didn't affect the result in any way
9. The first, more comprehensive model try-out was with KNN, for multiple neighbor settings and distance-based weighting (with varying distance metric as well), but the top performance was slightly under the top 2 up to this point (3, 7)
10. The next one was DecisionTreeClassifier, but no setup was able to improve upon 3 or 7
11. The next stop was SupportVectorClassifier with default settings, where SVM was able to improve upon 7
12. During SVM experiments, a regularization term of 0.9 revealed a setup of a not so good recall, but an outstanding precision
13. The final SVM experimentation was with the kernel function, where linear kernel surpassed my computational capabilities, but a polinomial kernel managed to produce the opposite of 12 when PCA and undersampling, but found a good trade-off when PCA and scaling
14. Finally concluded the SVM kernel experimentation with a sigmoidal kernel with a good recall value and moderately bad precision value when using PCA, scaling and resampling
15. The next one is logistic regression, where the model fails to converge without scaling, therefore logistic regression experiments will exclude setups without scaling
16. Logistic regression with or without undersampling manages to improve upon recall with slighter decrease in precision, and also sets record for F1
17. When using logistic regression with only PCA and scaling but without the default L2 penalty, it further improves F1
18. To conclude logistic regression, L1 penalty did not improve on the results, while elasticnet was only able to slightly improve on L2 results
19. The next step was quite short, since ensemble methods weren't able to improve upon the results of simple methods (Gradient Boosting and Random Forest from Sklearn and XGBoost libraries, and AdaBoost)
20. To conlcude modelling, the last resort was deep learning, an MLP, where MLP was able to catch up to the previous results, but at a high computational cost, therefore it will not be displayed in the final table

## Evaluation

When drawing up the baseline, a decision was made to do evaluation based on accuracy, recall, precision and F1 in order to measure not just the amount of hits, but to account for the imbalanced nature of the problem as well. The following table is an extract of this approach, denoting important milestones (ID numbers of experiments are taken from Experiment log):

| Experiment                                                           | Accuracy | Recall | Precision | F1     |
|----------------------------------------------------------------------|----------|--------|-----------|--------|
| I. KNN, closest neighbor, numericals (1)                             | 75.99%   | 25.33% | 25.04%    | 24.96% |
| II. Gradient boosting, numericals (3)                                | 84.42%   | 21.11% | 53.89%    | 30.02% |
| III. Gradient boosting, all features (3)                             | 87.14%   | 33.80% | 73.73%    | 44.33% |
| IV. _III._ + 15-PCA (4)                                              | 87.35%   | 33.88% | 74.06%    | 45.55% |
| V. _IV._ + RandomUnderSampler (5)                                    | 64.76%   | 66.27% | 26.40%    | 37.70% |
| VI. _V._ + MinMaxScaler (7)                                          | 72.79%   | 65.80% | 32.98%    | 43.79% |
| VII. SVM with MinMaxScaler, PCA and RandomUnderSampler (11)          | 75.92%   | 69.62% | 37.28%    | 48.42% |
| VIII. SVM with MinMaxScaler, PCA and regularization of 0.9 (12)      | 86.87%   | 21.16% | 91.71%    | 33.52% |
| IX. SVM with PCA and RandomUnderSampler, with polinomial kernel (13) | 29.46%   | 92.90% | 17.79%    | 29.82% |
| X. SVM with PCA and MinMaxScaler, with polinomial kernel (13)        | 85.51%   | 43.91% | 56.50%    | 49.24% |
| XI. _VII._ + with sigmoidal kernel (14)                              | 73.20%   | 72.95% | 34.41%    | 46.72% |
| XII. LogisticRegression with PCA and scaling (16)                    | 88.37%   | 38.01% | 79.90%    | 50.86% |
| XIII. _XII._ + RandomUnderSampler (16)                               | 74.01%   | 74.69% | 35.66%    | 48.16% |
| XIV. _XII._ without penalty term (17)                                | 88.03%   | 42.23% | 73.00%    | 52.62% |
| XV. _XII._ with ElasticNet penalty term (18)                         | 88.50%   | 38.86% | 79.92%    | 51.68% |

## Conclusions / Possible future steps

The task was to realize a binary classification model that is able to predict attrition rate at a company. The top results were not chosen based on accuracy, because this was an unbalanced problem, by predicting everything as negative class we would have been able to achieve 83.67% accuracy, which is quite convincing on its own, but the result is not usable. Therefore Recall, Precision and F1 were added to the measurements to determine the best 2 setups. 2 were chosen, because I wanted a result with high recall and low precision, one with low recall and high precision (so the 2 extreme points of the recall-precision tradeoff). The difference between the two was resampling. For this simple dataset simpler methods performed better, while ensemble methods and deep learning wasn't able to compete with these, because even if they were optimised to show the same or slightly better performance, their computational needs and challanges in terms of explainability would render them less useful. Let us look at the results from the evaluation table:
1. High recall, low precision: XIII. or IX.
2. Low recall, high precision: VIII. or XIV. or XV.

To answer which is the best one will need to further discuss with stakeholder in order to determine which has the higher cost, not finding those about to leave or producing false positives. The answer to this question would determine which to choose.

Possible future steps would mainly address two things:
1. Try-out of a broader palette of algorithms
2. More detailed discovery of the hyperparameter space, or even hyperparametertuning