# Credit Risk Modeling for Loan Defaults

## Project Overview
This project aims to measure the credit risk of a lending institution (i.e. a commercial bank) by calculating the expected loss of the outstanding loans. Credit risk is the likelihood that a borrower would not repay their loan to the lender. By measuring the risk effectively, a lender could minimize its credit losses while it reaches the fullest potential to maximize revenues on loan borrowing. It is also crucial for banks to abide by regulations that require them to conduct their business with sufficient capital adequacy, which, if in low, will risk the stability of the economic system.

The key metric of credit risk is Expected Loss (EL), calculated by multiplying the results across three models: PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default). The project includes all three models to help reach the final goal of credit risk measurement.
 
## Code and Resources Used
* __Python Version__: 3.8.5
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle
* __Algorithms__: regression (logistic, multiple linear)
* __Dataset Source__: https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling

## Datasets Information<br>
[_**'loan_data_2007_2014.csv'**_](https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling?select=loan_data_2007_2014.csv) contains the past data of all loans that we use to train and test our model<br>
[_**'loan_data_2015.csv'**_](https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling?select=loan_data_2015.csv) contains the current data we will implement the model to measure the risk<br>
[_**'loan_data_defaults.csv'**_](https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling?select=loan_data_defaults.csv) contains only the past data of all defaulted loans



_**Note**_: I embedded the findings and interpretations in the project-walkthrough below, and denoted them by 🔶

## [1. Data Preparation](https://github.com/shawn-y-sun/Credit_Risk_Model_LoanDefaults/blob/main/1.Credit%20Risk%20Modeling_PD%20Data%20Preparation.ipynb)

In this part of data pipeline, we fill in or convert the data into what we need, and then create and group dummy variables for each category as required for PD model and credit scorecard building.

### 1.1 Preprocessing Data
__Continuous variables__<br>
Convert the date string values to numeric values of days or months until today
- 'emp_length'
- 'term'
- 'earliest_cr_line'
- 'issue_d'
```
loan_data['emp_length_int'] = \
loan_data['emp_length'].str.replace('\+ years', '')
loan_data['emp_length_int'] = \
loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = \
loan_data['emp_length_int'].str.replace('n/a',  str(0))
loan_data['emp_length_int'] = \
loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = \
loan_data['emp_length_int'].str.replace(' year', '')
#Replace the text/blank strings to null; for example: '+ years','year', etc.
```
```
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])
# Transforms the values to numeric
```


#### Missing values
Replace with appropriate values or 0
```
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
# missing'total revolving high credit limit' = 'funded amount'

loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(),inplace = True)
# missing'annual income' = average of 'annual income'
```
```
loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)
# Other missing values = 0
```


### 1.2 Creating Dummy Variables<br>
For PD Model, we create dummy variables according to regulations to make the model easily understood and create credit scorecard


#### Dependent variable
We determine whether the loan is good (i.e. not defaulted) by looking at 'loan_status'. We assign a value of 1 if the loan is good, 0 if not
```
loan_data['good_bad'] = \
np.where(loan_data['loan_status'].\
         isin(['Charged Off', 'Default',
               'Does not meet the credit policy. Status:Charged Off',
               'Late (31-120 days)']), 0, 1)
```


#### Discrete Categories
```
loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':'),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':'),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':')] 
```


### 1.3 Grouping Dummy Variables
#### Methodology: 'Weight of Evidence' and 'Information Value'
- 'Weight of Evidence' shows to what extent an independent variable would predict a dependent variable, giving us an insight into how useful a given category of an independent variable is

WoE = ln(%good / %bad)

- Similarly, 'Information Value', ranging from 0 to 1, shows how much information the original independent variable brings with respect to explaining the dependent variable, helping to pre-select a few best predictors

IV = Sum((%good - %bad) * WoE)

| Range: 0-1      | Predictive powers                       |
|-----------------|-----------------------------------------|
| IV < 0.02       | No power                                |
| 0.02 < IV < 0.1 | Weak power                              |
| 0.1 < IV < 0.3  | Medium power                            |
| 0.3 < IV < 0.5  | Strong power                            |
| 0.5 < IV        | Suspiciously high, too good to be true |


#### Creating WoE and Visualization Function
```
# WoE function for discrete unordered variables
def woe_discrete(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    # Only store the independent and dependent variables
    
    df = pd.concat([df.groupby(df.columns.values[0], 
                               as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], 
                               as_index = False)[df.columns.values[1]].mean()], 
                   axis = 1)
    # Group the df by value in first column
    # Add the number and mean of good_bad obs of each kind
    
    df = df.iloc[:, [0, 1, 3]]
    # Remove the 3rd column
    
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    # Rename the columns
    
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    # Calculate proportions of each kind
    
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
     # Calculate the number of good and bad borrowers of each kind
        
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    # Calculate the proportions of good and bad borrowers of each kind
    
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    # Calculate the weight of evidence of each kind
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    # Sort the dataframes by WoE, and replace the index with increasing numbers
    
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    # Calculate the difference of certain variable between the nearby kinds
    
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    #Calculatet he information value of each kind
    
    return df
# The function takes 3 arguments: a dataframe, a string, and a dataframe. 
#  The function returns a dataframe as a result.
```

```
def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    # Turns the values of the first column to strings, 
    #   makes an array from these strings, and passes it to variable x
    y = df_WoE['WoE']
    # Selects a column with label 'WoE' and passes it to variable y
    plt.figure(figsize=(18, 6))
    # Sets the graph size to width 18 x height 6.
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    # Plots the datapoints with coordiantes variable x on the x-axis 
    #  and variable y on the y-axis
    # Sets the marker for each datapoint to a circle, 
    #  the style line between the points to dashed, and the color to black
    plt.xlabel(df_WoE.columns[0])
    # Names the x-axis with the name of the first column
    plt.ylabel('Weight of Evidence')
    # Names the y-axis 'Weight of Evidence'.
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    # Names the grapth 'Weight of Evidence by ' the name of the column with index 0.
    plt.xticks(rotation = rotation_of_x_axis_labels)
    # Rotates the labels of the x-axis a predefined number of degrees
```


#### Computing and Visualizing WoE

For discrete variables, we order them by WoE and set the category with the worst credit risk as a reference category

|   | home_ownership | n_obs | prop_good | prop_n_obs | n_good | n_bad | prop_n_good | prop_n_bad | WoE       | diff_prop_good | diff_WoE | IV       |
|---|----------------|-------|-----------|------------|--------|-------|-------------|------------|-----------|----------------|----------|----------|
| 0 | OTHER          | 45    | 0.777778  | 0.000483   | 35     | 10    | 0.000421    | 0.000981   | -0.845478 | NaN            | NaN      | 0.022938 |
| 1 | NONE           | 10    | 0.8       | 0.000107   | 8      | 2     | 0.000096    | 0.000196   | -0.711946 | 0.022222       | 0.133531 | 0.022938 |
| 2 | RENT           | 37874 | 0.874003  | 0.406125   | 33102  | 4772  | 0.398498    | 0.468302   | -0.161412 | 0.074003       | 0.550534 | 0.022938 |
| 3 | OWN            | 8409  | 0.888572  | 0.09017    | 7472   | 937   | 0.089951    | 0.091953   | -0.022006 | 0.014568       | 0.139406 | 0.022938 |
| 4 | MORTGAGE       | 46919 | 0.904751  | 0.503115   | 42450  | 4469  | 0.511033    | 0.438567   | 0.152922  | 0.016179       | 0.174928 | 0.022938 |

![image](https://user-images.githubusercontent.com/77659538/110436112-c88a2300-80ee-11eb-979c-958f33acc1ea.png)


For continuous variables, we put them in bins of same size (fine classing) and set the minimal bin as the reference category
```
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_acc_factor', df_targets_prepr)
df_temp.head()
```

|   | total_acc_factor | n_obs | prop_good | prop_n_obs | n_good | n_bad | prop_n_good | prop_n_bad | WoE       | diff_prop_good | diff_WoE | IV  |
|---|------------------|-------|-----------|------------|--------|-------|-------------|------------|-----------|----------------|----------|-----|
| 0 | (-0.156, 3.12]   | 125   | 0.776     | 0.00134    | 97     | 28    | 0.001168    | 0.002748   | -0.855734 | NaN            | NaN      | inf |
| 1 | (3.12, 6.24]     | 1499  | 0.850567  | 0.016074   | 1275   | 224   | 0.015349    | 0.021982   | -0.359185 | 0.074567       | 0.496549 | inf |
| 2 | (6.24, 9.36]     | 3715  | 0.871871  | 0.039836   | 3239   | 476   | 0.038993    | 0.046712   | -0.180639 | 0.021304       | 0.178547 | inf |
| 3 | (9.36, 12.48]    | 6288  | 0.874841  | 0.067427   | 5501   | 787   | 0.066224    | 0.077233   | -0.153784 | 0.00297        | 0.026855 | inf |
| 4 | (12.48, 15.6]    | 8289  | 0.888286  | 0.088883   | 7363   | 926   | 0.088639    | 0.090873   | -0.024892 | 0.013445       | 0.128892 | inf |


#### Grouping Categories

We group the categories or bins by following features:
- Small number of observations
- Similar WoE
- Outliers

```
# 'OTHERS' and 'NONE' are riskiest but are very few
# 'RENT' is the next riskiest.
# 'ANY' are least risky but are too few. 
# -> Conceptually, they belong to the same category. 
#    Also, their inclusion would not change anything.
# -> We combine them in one category, 'RENT_OTHER_NONE_ANY'.
#    We end up with 3 categories: 'RENT_OTHER_NONE_ANY', 'OWN', 'MORTGAGE'.

df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = \
sum([df_inputs_prepr['home_ownership:RENT'], 
     df_inputs_prepr['home_ownership:OTHER'], 
     df_inputs_prepr['home_ownership:NONE'],
     df_inputs_prepr['home_ownership:ANY']])
```

```
# Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = \
np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = \
np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = \
np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)
```

### 1.4 Create and Export Train and Test Datasets

#### Before Grouping Dummies' Pipeline

We first split the dataset into training and testing parts
```
loan_data_inputs_train, loan_data_inputs_test, \
loan_data_targets_train, loan_data_targets_test = \
train_test_split(loan_data.drop('good_bad', axis = 1), \
                 loan_data['good_bad'], test_size = 0.2, random_state = 42)
# Split two dataframes with inputs and targets, 
#   each into a train and test dataframe, and store them in variables.
# Set the size of the test dataset to be 20%.

# Set a specific random state.
#  Allow us to perform the exact same split multiple times.
#  To assign the exact same observations to the train and test datasets.
```

We assign the train or test datasets to the 'Grouping Dummies' pipeline
```
#####
df_inputs_prepr = loan_data_inputs_train
df_targets_prepr = loan_data_targets_train
#####
#df_inputs_prepr = loan_data_inputs_test
#df_targets_prepr = loan_data_targets_test
```


#### After Grouping Dummies' Pipeline

We store the dataset with grouped dummies in a to-be-saved dataset
```
#####
loan_data_inputs_train = df_inputs_prepr
#####
#loan_data_inputs_test = df_inputs_prepr
```

We save the preprocessed datasets to CSV
```
loan_data_inputs_train.to_csv('loan_data_inputs_train.csv')
loan_data_targets_train.to_csv('loan_data_targets_train.csv')
loan_data_inputs_test.to_csv('loan_data_inputs_test.csv')
loan_data_targets_test.to_csv('loan_data_targets_test.csv')
```

## [2. PD Model Building](https://github.com/shawn-y-sun/Credit_Risk_Model_LoanDefaults/blob/main/2.Credit%20Risk%20Modeling_PD%20Model%20Building.ipynb)


### 2.1 Model Building
We will trian and fit statistical model to predict a borrower's probability of being good 

#### Excluding Features

We exclude the reference categories in our model as we used dummy variables to represent all features, making the reference categories redundant

#### Choosing Logistic Regression

We select logistic regression as our model because the outcome variable has only two outcomes: good (1) or bad (0)

#### Building Logistic Regression with P-Values
```
# As there is no built-in method to calculate P values for 
#  sklearn logistic regression

# Build a Class to display p-values for logistic regression in sklearn.

from sklearn import linear_model
import scipy.stats as stat

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values
```

#### Selecting Features

We leave a group of features if one of the dummy variables has a p value smaller than 0.05, meaning it is a significant variable.

#### Training the Model with Selected Features
```
reg2 = LogisticRegression_with_p_values()
reg2.fit(inputs_train, loan_data_targets_train)
```

We get the coefficients of each significant variable

| Feature name | Coefficients                 | P Values  |          |
|--------------|------------------------------|-----------|----------|
| 0            | Intercept                    | -1.374036 | NaN      |
| 1            | grade:A                      | 1.123662  | 3.23E-35 |
| 2            | grade:B                      | 0.878918  | 4.28E-47 |
| 3            | grade:C                      | 0.684796  | 6.71E-34 |
| 4            | grade:D                      | 0.496923  | 1.35E-20 |
| ...          | ...                          | ...       | ...      |
| 80           | mths_since_last_record:3-20  | 0.440625  | 3.35E-04 |
| 81           | mths_since_last_record:21-31 | 0.35071   | 1.83E-03 |
| 82           | mths_since_last_record:32-80 | 0.502956  | 3.96E-09 |
| 83           | mths_since_last_record:81-86 | 0.175839  | 8.60E-02 |
| 84           | mths_since_last_record:>86   | 0.232707  | 5.71E-03 |

🔶 Interpretation of the coefficients β: the odds for someone being a good borrower is e^(β) times higher than the odds for someone with the reference feature<br>
(Note: direct comparison are possible only between categories coming from one and the same original independent variable)

Finally, we save the model
```
pickle.dump(reg2, open('pd_model.sav', 'wb'))
```

### 2.2 Model Evaluation
We will apply the model on our testing dataset and use different measures to assess how accurate our model is, to know to what extent the outcome of interest can be explained by the available information.

#### Excluding Features

We remove the reference categories and insignificant features


#### Testing Model

After running model on testing dataset, we get the following results

|               |     loan_data_targets_test    |     y_hat_test_proba    |
|---------------|-------------------------------|-------------------------|
|     362514    |     1                         |     0.924306            |
|     288564    |     1                         |     0.849239            |
|     213591    |     1                         |     0.885349            |
|     263083    |     1                         |     0.940636            |
|     165001    |     1                         |     0.968665            |

🔶 'y_hat_test_proba': it tells a borrower's probability of being a good borrower (won't default)


#### Confusion Matrix

We set the threshold at 0.9, meaning the borrower is predicted to a good borrower if the y_hat_test_proba >= 0.9
```
tr = 0.9
df_actual_predicted_probs['y_hat_test'] = \
np.where(df_actual_predicted_probs['y_hat_test_proba'] >= tr, 1,0)
```


Then we can create the confusion matrix

| Predicted/ Actual | 0        | 1        |
|-------------------|----------|----------|
| 0                 | 0.079072 | 0.030196 |
| 1                 | 0.384025 | 0.506707 |

🔶 When we set the threshold at 0.9, it would reduce the number of defaults, but also the number of overall approved loans. It means we can avoid much of the credit risk but cannot reach the best potential to make money (~38% of borrowers will be mistakenly rejected).

```
In [50]:
true_neg = confusion_matrix_per.iloc[0,0]
true_pos = confusion_matrix_per.iloc[1,1]
true_rate = true_neg + true_pos
true_rate

Out[50]:
0.5857790836076648
```
🔶 The result implies our model can correctly classify ~59% of borrowers when we set threshold at 0.9. It means our model has some predicting power since the it is greater than 50%, which is the accuracy of predicting by chance. But the power still is not strong enough. However, if we set the threshold at different levels, the accuracy will vary accordingly. Thus we need other measures to evaluate our model.

#### ROC Curve and AUC

We further assess the predicting power by plotting the true positive rate against the false positive rate at various threshold settings

ROC plot<br>
![image](https://user-images.githubusercontent.com/77659538/110450038-c7142700-80fd-11eb-945b-271244e47843.png)

🔶 Each point from the curve represents a different confusion matrix based on a different threshold. In specific, it is equal to (True positive rate / False positive rate)

We compute the area under ROC (AUROC)<br>
```
In [56]:
AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], 
                      df_actual_predicted_probs['y_hat_test_proba'])
AUROC

Out[56]:
0.702208104993648
```
🔶 The result implies our model has a 'fair' predicting power, not perfect but good enough to use.


#### Gini and Kolmogorov-Smirnov

Gini coefficient measures the inequality between good borrowers and bad borrowers in a population.

```
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'],
         df_actual_predicted_probs['Cumulative Perc Bad'])
plt.plot(df_actual_predicted_probs['Cumulative Perc Bad'],
         df_actual_predicted_probs['Cumulative Perc Bad'],
        linestyle = '--', color ='k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
```

![image](https://user-images.githubusercontent.com/77659538/110452593-4acf1300-8100-11eb-955c-e8885aaa5893.png)

```
In [61]:
Gini = AUROC * 2 - 1
Gini

Out[61]:
0.4044162099872961
```

🔶 The curve demonstrates the model's accuracy of recognizing bad borrowers as the threshold increased and more borrowers are rejected. For example, when we reject 20% of the borrowers based on our model, about 40% of the bad borrowers will be rejected, meaning we have a higher predicting power by rejecting them by chance.<br>
🔶 The curve is upward further apart from the diagonal line and the model has satisfactory predictive power.


Kolmogorov-Smirnov coefficient measures the maximum difference between the cumulative distribution functions of 'good' and 'bad' borrowers. 

```
plt.plot(df_actual_predicted_probs['y_hat_test_proba'],
         df_actual_predicted_probs['Cumulative Perc Bad'])
plt.plot(df_actual_predicted_probs['y_hat_test_proba'],
         df_actual_predicted_probs['Cumulative Perc Good'],
        linestyle = '--', color ='k')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
```

![image](https://user-images.githubusercontent.com/77659538/110452594-4acf1300-8100-11eb-8faa-c5f98c53c0cc.png)


```
In [63]:
KS = max(df_actual_predicted_probs['Cumulative Perc Bad'] - df_actual_predicted_probs['Cumulative Perc Good'])
KS

Out[63]:
0.2966746932223847
```
🔶 The plot demonstrates the percentage of bad or good borrowers will be rejected at various threshold settings. For example, if we set threshold at 0.8, ~25% of the bad borrowers will be rejected while only ~10% of good borrowers will be rejected.<br>
🔶 The two cumulative distribution functions are sufficiently far away from each other and the model has satisfactory predictive power.

### 2.3 Credit Scorecard Building ([Full Scorecard](https://github.com/shawn-y-sun/Credit_Risk_Model_LoanDefaults/blob/main/df_scorecard.csv))

#### Preprocessing the coefficient tables
1. Add back the reference categories and assign them coefficients of 0
2. Determine original feature name for each sub feature
3. Determine the minimum and maximum coefficients for each original feature name
4. Calculate the sum of minimum and maximum coefficients to determine the weight for lowest and highest possible weight a borrower could have

#### Convert Coefficients to FICO Scores
To calculate the score for each feature<br>
![image](https://user-images.githubusercontent.com/77659538/110462984-54f70e80-810c-11eb-8555-0953fd44ac12.png)

To calculate the score for the intercept<br>
![image](https://user-images.githubusercontent.com/77659538/110463011-5a545900-810c-11eb-9de3-054e8ab56d3d.png)


#### Findings on Scorecard
If we rank the scorecard by scores<br>
|    | Feature name             | Score - Final |
|----|--------------------------|---------------|
| 0  | Intercept                | 313           |
| 1  | grade:A                  | 87            |
| 35 | mths_since_issue_d:<38   | 84            |
| 2  | grade:B                  | 68            |
| 36 | mths_since_issue_d:38-39 | 68            |

|    | Feature name                        | Score - Final |
|----|-------------------------------------|---------------|
| 41 | mths_since_issue_d:65-84            | -6            |
| 55 | annual_inc:20K-30K                  | -6            |
| 23 | verification_status:Source Verified | -1            |
| 56 | annual_inc:30K-40K                  | -1            |
| 85 | grade:G                             | 0             |

🔶 A external rating of A adds most value to a borrower's credit score, followed by if the months since issue date is fewer than 38 days. <br>
🔶 A person's credit score can be most negatively impacted if his annual income is between 20K and 30K and the loan has been issued for 65-84 months


Then we look at some features in detail, we begin with annual income<br>
| Feature name         | Score - Final |
|----------------------|---------------|
| annual_inc:120K-140K | 43            |
| annual_inc:>140K     | 38            |
| annual_inc:100K-120K | 36            |
| annual_inc:90K-100K  | 30            |
| annual_inc:80K-90K   | 28            |
| annual_inc:70K-80K   | 22            |
| annual_inc:60K-70K   | 17            |
| annual_inc:50K-60K   | 11            |
| annual_inc:40K-50K   | 6             |
| annual_inc:<20K      | 0             |
| annual_inc:30K-40K   | -1            |
| annual_inc:20K-30K   | -6            |

🔶 Not surprisingly, scores are positively related with annual income and each level of income does differentiates the score by a lot

Then look at which state the borrower lives<br>
| Feature name                    | Score - Final |
|---------------------------------|---------------|
| addr_state:WV_NH_WY_DC_ME_ID    | 40            |
| addr_state:KS_SC_CO_VT_AK_MS    | 25            |
| addr_state:IL_CT                | 20            |
| addr_state:WI_MT                | 18            |
| addr_state:TX                   | 17            |
| addr_state:GA_WA_OR             | 14            |
| addr_state:AR_MI_PA_OH_MN       | 10            |
| addr_state:RI_MA_DE_SD_IN       | 8             |
| addr_state:UT_KY_AZ_NJ          | 6             |
| addr_state:CA                   | 5             |
| addr_state:NY                   | 4             |
| addr_state:OK_TN_MO_LA_MD_NC    | 4             |
| addr_state:NM_VA                | 3             |
| addr_state:ND_NE_IA_NV_FL_HI_AL | 0             |

🔶 Where a person lives is also a differentiator: borrowers living in West Virginia, New Hampshire, Wyoming, DC, Maine, Idaho are much more likely to have a higher credit score than borrowers living in other states. But considering the population of these states are relatively smaller and thus have a smaller number of observations in our dataset, there might be some bias.

Finally, we look at employment length<br>
| Feature name   | Score - Final |
|----------------|---------------|
| emp_length:2-4 | 10            |
| emp_length:10  | 10            |
| emp_length:1   | 8             |
| emp_length:5-6 | 7             |
| emp_length:7-9 | 5             |
| emp_length:0   | 0             |

🔶 Surprisingly, looks like employment length is negatively related to a person's credit score. It is possibly because a young worker does not have debt and spending on family, thus they face less financial stress and have a smaller chance to default on loans. Another reason could be that fewer young workers have been approved a loan thus we don't have enough of their data.


## [3. PD Model Monitoring](https://github.com/shawn-y-sun/Credit_Risk_Model_LoanDefaults/blob/main/3.Credit%20Risk%20Modeling_PD%20Model%20Monitoring.ipynb)
This part will assess if the PD model is out-of-date and needs to be re-trained by the newest dataset. We use PSI (Population Stability Index) to measure how much the variables has shifted over time. A high PSI indicates that the overall characteristics of borrowers have changed, meaning our model might not fit the new population as well as before, therefore it needs to be updated.

## [4. LGD & EAD Model Building](https://github.com/shawn-y-sun/Credit_Risk_Model_LoanDefaults/blob/main/4.Credit%20Risk%20Modeling_LGD%20%26%20EAD%20Model.ipynb)
In this part, we choose appropriate statistical models (linear/logistic regression) to train the LGD and EAD models, and we trained them using the dataset including only defaulted borrowers. The data preprocessing and model building approaches are quite similar to what we have done in PD Model.

Finally, we combined three models (PD, LGD, and EAD) together to calculate the expected loss. 
```
In [109]:
ratio_EL = total_EL / total_funded
ratio_EL

Out[109]:
0.07526562017218118
```
🔶 The ratio of expected loss over total funded amount is 7.5%, which is an acceptable level and means our credit risk is under control!
