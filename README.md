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

### Creating Dummy Variables <br>
For PD Model, We create dummy variables according to regulations to make the model easily understood and create credit scorecard.

__Dependent variable__<br>
We determine whether the loan is good (i.e. not defaulted) by looking at 'loan_status'. We assign a value of 1 if the loan is good, 0 if not.
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

### Calculating 'Weight of Evidence' and 'Information Value'<br>
__Methodology: 'Weight of Evidence' and 'Information Value'__<br>
We calculate 'Weight of Evidence' and 'Information Value' 
- 'Weight of Evidence' shows to what extent an independent variable would predict a dependent variable, giving us an insight into how useful a given category of an independent variable is. 

WoE = ln(%good / %bad)

- Similarly, 'Information Value', ranging from 0 to 1,  shows how much information the original independent variable brings with respect to explaining the dependent variable, helping to pre-select a few best predictors.

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

|   | home_ownership | n_obs | prop_good | prop_n_obs | n_good | n_bad | prop_n_good | prop_n_bad | WoE       | diff_prop_good | diff_WoE | IV       |
|---|----------------|-------|-----------|------------|--------|-------|-------------|------------|-----------|----------------|----------|----------|
| 0 | OTHER          | 45    | 0.777778  | 0.000483   | 35     | 10    | 0.000421    | 0.000981   | -0.845478 | NaN            | NaN      | 0.022938 |
| 1 | NONE           | 10    | 0.8       | 0.000107   | 8      | 2     | 0.000096    | 0.000196   | -0.711946 | 0.022222       | 0.133531 | 0.022938 |
| 2 | RENT           | 37874 | 0.874003  | 0.406125   | 33102  | 4772  | 0.398498    | 0.468302   | -0.161412 | 0.074003       | 0.550534 | 0.022938 |
| 3 | OWN            | 8409  | 0.888572  | 0.09017    | 7472   | 937   | 0.089951    | 0.091953   | -0.022006 | 0.014568       | 0.139406 | 0.022938 |
| 4 | MORTGAGE       | 46919 | 0.904751  | 0.503115   | 42450  | 4469  | 0.511033    | 0.438567   | 0.152922  | 0.016179       | 0.174928 | 0.022938 |

![image](https://user-images.githubusercontent.com/77659538/110436112-c88a2300-80ee-11eb-979c-958f33acc1ea.png)

__Combining Categories__<br>
We set the category with the worst credit risk as a reference category, then combine the categories with similar WoE for to simplify our model.
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




## Probability of Default Model (PD)
### Data Cleaning
Dataset: *'loan_data_2007_2014.csv'* <br>
* Correct the invalid dates
* Convert the relevant dates to useful number of days
* Covert data strings to numerical values
* Fill the missing values

### Data Preparation
* Create dummy variables
* Determine the Weight of Evidence (WoE)
* For categorical variables, order them by WoE
* For continuous variables, group adjacent duummy variables based on WoE
* Remove the reference variable

### Model Building
__Algorithm:__ Logistic Regression <br>
__Outcome Variable:__ *loan_status* <br>

### Model Evaluation
__Confusion Matrix__

| Predicted<br>Actual | 0     | 1        |
|------------------|----------|----------|
| 0                | 0.079072 | 0.030196 |
| 1                | 0.384025 | 0.506707 |

True Rate = 0.5857790836076648


__ROC Curve & AUC__<br>
![image](https://user-images.githubusercontent.com/77659538/109492048-564d8900-7ac5-11eb-8ba7-321976cef573.png)<br>
AUROC = 0.702208104993648


__Gini Coefficient & Kolmogorov-Smirnov__<br>
![image](https://user-images.githubusercontent.com/77659538/109492110-6d8c7680-7ac5-11eb-9844-ad4ba43aee27.png)<br>
Gini = 0.4044162099872961

![image](https://user-images.githubusercontent.com/77659538/109492134-754c1b00-7ac5-11eb-9843-472ce812e8e1.png)<br>
Kolmogorov-Smirnov = 0.2966746932223847

### Model Monitoring
Indicator: Population Stability Index (PSI)


## Loss Given Default Model (LGD)
__Dataset:__ _'loan_data_defaults.csv'_ <br>
__Outcome Variable:__ *recovery_rate* <br>
__Algorithm:__ Logistic Regression (stage 1), Linear Regression (stage 2)
  - Stage 1: determine if recovery rate is 0 or not
  - Stage 2: if not 0, how much is the recovery rate

## Exposure at Default Model (EAD)
__Dataset:__ _'loan_data_defaults.csv'_ <br>
__Outcome Variable:__ *CCF* (credit conversion factor) <br>
__Algorithm:__ Linear Regression

## Expected Loss (EL) and Conclusion
* Combine the three models (PD, LGD, EDA) to get the expected loss
* (EL / total loan amount) < 10% -> meet the requirement (credit risk is under control)




