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

### Feature selection

## Pattern search (modelling)

The modelling approaches used here are broken down into three categories from explainability and simplicity point of view.

### Base models

### Ensemble models

### Deep learning

## Evaluation