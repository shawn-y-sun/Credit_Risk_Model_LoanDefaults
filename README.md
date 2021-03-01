# Credit Risk Modeling for Loan Defaults

## Project Overview
* Goal: this project aims to measure the credit risk of a lending institution by calculating the expected loss of the outstanding loans
* Reasons: credit risk modeling is part of the regulartory requirements of risk management for financial institutions; it also allows a loan-lending practice to maximize profits while having the risk under control
* Approach: the Expected Loss (EL) are calculated by multiplying three credit risk models, PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default)

## Code and Resources Used
* __Python Version__: 3.8.5
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle
* __Dataset Source__: https://www.kaggle.com/shawnysun/loan-data-for-credit-risk-modeling

## Datasets Information
*'loan_data_2007_2014.csv'* <br>
_'loan_data_2015.csv'_ <br>
_'loan_data_defaults.csv'_ <br>

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




