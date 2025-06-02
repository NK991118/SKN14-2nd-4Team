import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


original = pd.read_csv('./data/HR_Employee_Attrition.csv')

filtered_columns = ['Age', 'Attrition', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']
object_columns = ['Attrition', 'MaritalStatus', 'OverTime']
data = original[filtered_columns]


column_mappings = {}

for column in object_columns:
    mapping = {}
    if column == 'Attrition' or column == 'OverTime':
        mapping['Yes'] = 1
        mapping['No'] = 0
    else:
        for idx, content in enumerate(data[column].unique()):
            mapping[content] = idx
    
    column_mappings[column] = mapping
    data[column] = data[column].map(column_mappings[column])

y = data['Attrition']
X = data.drop(labels='Attrition', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

train = pd.concat([X_train, y_train], axis=1, ignore_index=False, sort=False)
test = pd.concat([X_test, y_test], axis=1, ignore_index=False, sort=False)
        
train.to_csv('./data/train.csv', index=False)
test.to_csv('./data/test.csv', index=False)


"""
column_mappings = {
'Attrition': {'No': 0, 'Yes': 1}, 
'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2}, 
'OverTime': {'No': 0, 'Yes': 1}
}
"""