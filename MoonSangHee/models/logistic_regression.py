import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer


data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

X = data.drop('Attrition', axis=1)#.to_numpy()
y = data['Attrition']#.to_numpy()

X_test = test_data.drop('Attrition', axis=1)#.to_numpy()
y_test = test_data['Attrition']#.to_numpy()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 예시로 컬럼 나눠보자
numeric_features = ['Age', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']
categorical_features = ['MaritalStatus', 'OverTime']

# 1. 숫자형 컬럼은 StandardScaler
# 2. 범주형 컬럼은 그대로 (혹은 필요하면 OneHotEncoder)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)  # 범주형은 손대지 않고 넘긴다
    ]
)

# scaler = StandardScaler()
X_train_scaled = preprocessor .fit_transform(X_train)
X_val_scaled =  preprocessor .transform(X_val)
X_test_scaled = preprocessor .transform(X_test)

smote = SMOTE(random_state=42, sampling_strategy = 0.6)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# lr_clf = LogisticRegression()
# lr_clf = LogisticRegression(C = 0.39069, solver='newton-cg', class_weight='balanced')
# lr_clf = LogisticRegression(solver='liblinear', penalty='l1')
# lr_clf = LogisticRegression(class_weight = {0:1, 1 : 2})
# lr_clf = LogisticRegression(class_weight='balanced')
# lr_clf = XGBClassifier(objective='binary:logistic')
# lr_clf = LGBMClassifier(verbosity = 1)
# lr_clf = KNeighborsClassifier()
# lr_clf = MLPClassifier()
# lr_clf = SVC()
lr_clf = RandomForestClassifier(max_depth=15, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
# lr_clf = DecisionTreeClassifier()
# lr_clf = VotingClassifier()

# lr_clf.fit(X_train_scaled, y_train)
lr_clf.fit(X_train_resampled, y_train_resampled)


# threshold = 0.45 
# y_pred = (lr_clf.predict_proba(X_val)[:, 0] >= threshold).astype(int)

y_pred = lr_clf.predict(X_val_scaled)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# print('train_accuracy: ', lr_clf.score(X_train_scaled, y_train))
print('train_accuracy: ', lr_clf.score(X_train_resampled, y_train_resampled))
print('val_accuracy: ', lr_clf.score(X_val_scaled, y_val))

print(classification_report(y_val, y_pred))
print('AUC-ROC:', roc_auc_score(y_val, y_pred))

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')

print('===============test==================')

y_test_pred = lr_clf.predict(X_test_scaled)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print('test_accuracy: ', lr_clf.score(X_test_scaled, y_test))
print(classification_report(y_test, y_test_pred))
print('AUC-ROC:', roc_auc_score(y_test, y_test_pred))

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')
