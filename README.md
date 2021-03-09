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
Dates variables: convert it to numeric values of days / months until today
- emp_length
- term
- earliest_cr_line 
- issue_d

### Creating Dummy Variables
For PD Model, We create dummy variables according to regulations to make the model easily understood and create credit scorecard.

__Discrete variables__<br>
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




