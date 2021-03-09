# Credit Risk Modeling for Loan Defaults

## Project Overview
This project aims to measure the credit risk of a lending institution (i.e. a commerical bank) by calculating the expected loss of the outstanding loans. Credit risk is the likelihood that a borrower would not repay their loan to the lender. By measuring the risk effectively, a lender could minimize its credit losses while it reaches the fullest potential to maximize revenues on loan borrowing. It is also crucial for banks to abide by regulations that require them to conduct their business with sufficent capital adequacy, which, if in low, will risk the stability of the economic system.

The key metric of credit risk is Expected Loss (EL), calculated by multiplying the results across three models: PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default). The project includes all three models to help reach the final goal of credit risk measurement.
 
## Code and Resources Used
* __Python Version__: 3.8.5
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle
* __Dataset Source__: https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling

## Datasets Information<br>
[_**'loan_data_2007_2014.csv'**_](https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling?select=loan_data_2007_2014.csv) contains the past data of all loans that we use to train and test our model
[_**'loan_data_2015.csv'**_](https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling?select=loan_data_2015.csv) contains the current data we will implement the model to measure the risk
[_**'loan_data_defaults.csv'**_](https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling?select=loan_data_defaults.csv) contains only the past data of all defaulted loans

## [1. Data Preparation](https://github.com/shawn-y-sun/Credit_Risk_Model_LoanDefaults/blob/main/1.Credit%20Risk%20Modeling_PD%20Data%20Preparation.ipynb)

In this part of data pipeline, we fill in or convert the data into what we need, and then create and group dummy variables for each category as required for PD model and credit scorecard building.

### Preprocessing Data
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


__Missing values__<br>
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


### Creating Dummy Variables<br>
For PD Model, We create dummy variables according to regulations to make the model easily understood and create credit scorecard


__Dependent variable__<br>
We determine whether the loan is good (i.e. not defaulted) by looking at 'loan_status'. We assign a value of 1 if the loan is good, 0 if not
```
loan_data['good_bad'] = \
np.where(loan_data['loan_status'].\
         isin(['Charged Off', 'Default',
               'Does not meet the credit policy. Status:Charged Off',
               'Late (31-120 days)']), 0, 1)
```


__Discrete Categories__<br>
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


### Grouping Dummy Variables
__Methodology: 'Weight of Evidence' and 'Information Value'__<br>
- 'Weight of Evidence' shows to what extent an independent variable would predict a dependent variable, giving us an insight into how useful a given category of an independent variable is

WoE = ln(%good / %bad)

- Similarly, 'Information Value', ranging from 0 to 1,  shows how much information the original independent variable brings with respect to explaining the dependent variable, helping to pre-select a few best predictors

IV = Sum((%good - %bad) * WoE)

| Range: 0-1      | Predictive powers                       |
|-----------------|-----------------------------------------|
| IV < 0.02       | No power                                |
| 0.02 < IV < 0.1 | Weak power                              |
| 0.1 < IV < 0.3  | Medium power                            |
| 0.3 < IV < 0.5  | Strong power                            |
| 0.5 < IV        | Suspisciously high, too good to be true |


__Creating WoE and Visulization Function__

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


__Computing and Visualizing WoE__

For discrete variables, we order them by WoE and set the category with the worst credit risk as a reference category

|   | home_ownership | n_obs | prop_good | prop_n_obs | n_good | n_bad | prop_n_good | prop_n_bad | WoE       | diff_prop_good | diff_WoE | IV       |
|---|----------------|-------|-----------|------------|--------|-------|-------------|------------|-----------|----------------|----------|----------|
| 0 | OTHER          | 45    | 0.777778  | 0.000483   | 35     | 10    | 0.000421    | 0.000981   | -0.845478 | NaN            | NaN      | 0.022938 |
| 1 | NONE           | 10    | 0.8       | 0.000107   | 8      | 2     | 0.000096    | 0.000196   | -0.711946 | 0.022222       | 0.133531 | 0.022938 |
| 2 | RENT           | 37874 | 0.874003  | 0.406125   | 33102  | 4772  | 0.398498    | 0.468302   | -0.161412 | 0.074003       | 0.550534 | 0.022938 |
| 3 | OWN            | 8409  | 0.888572  | 0.09017    | 7472   | 937   | 0.089951    | 0.091953   | -0.022006 | 0.014568       | 0.139406 | 0.022938 |
| 4 | MORTGAGE       | 46919 | 0.904751  | 0.503115   | 42450  | 4469  | 0.511033    | 0.438567   | 0.152922  | 0.016179       | 0.174928 | 0.022938 |

![image](https://user-images.githubusercontent.com/77659538/110436112-c88a2300-80ee-11eb-979c-958f33acc1ea.png)


For continuous variables, we put them in a specifc number of bins and set the minimal bin as the reference category
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


__Grouping Categories__<br>

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

### Create and Export Train and Test Datasets

__Before Grouping Dummies' Pipeline__

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
#  Allow us to perform the exact same split multimple times.
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


__After Grouping Dummies' Pipeline__

We store the dataset with grouped dummies to a to_be_saved dataset
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


### Model Building

__Excluding Features__

We exclude the reference categories in our model as we used dummy variables to represent all features, making the reference categories redundant

__Choosing Logistic Regression__

We select logistic regression as our model because the outcome variable has only two outcomes: good (1) or bad (0)

__Building Logistic Regression with P-Values__
```
# As there is no built-in method to calcualte P values for 
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

__Selecting Features__

We leave a group of features if one of the dummy variables has a p value smaller than 0.05, meaning it is a significant variable

__Training the Model with Selected Features__
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

Finally, we save the model
```
pickle.dump(reg2, open('pd_model.sav', 'wb'))
```

### Model Validation
__Excluding Features__

We remove the reference categories and insignificant features

__Testing Model__

After running model on testing dataset, we get the following results

|               |     loan_data_targets_test    |     y_hat_test_proba    |
|---------------|-------------------------------|-------------------------|
|     362514    |     1                         |     0.924306            |
|     288564    |     1                         |     0.849239            |
|     213591    |     1                         |     0.885349            |
|     263083    |     1                         |     0.940636            |
|     165001    |     1                         |     0.968665            |

'y_hat_test_proba': it tells a borrower's probability of being a good borrower (won't default)

__Confusion Matrix__

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


```
In [50]:
true_neg = confusion_matrix_per.iloc[0,0]
true_pos = confusion_matrix_per.iloc[1,1]
true_rate = true_neg + true_pos
true_rate

Out[50]:
0.5857790836076648
```
