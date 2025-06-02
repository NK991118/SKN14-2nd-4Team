import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer

# 1. 데이터 준비 (예시)
data = pd.read_csv('./data/train.csv')
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# 2. Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 3. 스케일링 (클러스터링은 거리 기반이라 꼭 필요해)

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

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# 4. KMeans 클러스터링
# 클러스터 수(k)는 보통 3~10 정도로 시도해보면 좋아
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels_train = kmeans.fit_predict(X_train_scaled)

# 5. Train 데이터에 cluster feature 추가
X_train_with_cluster = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_with_cluster['cluster'] = cluster_labels_train
print('X_train_with_cluster[cluster]', X_train_with_cluster.groupby('cluster').count())

# 6. Test 데이터에도 같은 KMeans로 클러스터 예측
cluster_labels_test = kmeans.predict(X_test_scaled)
X_test_with_cluster = pd.DataFrame(X_test_scaled, columns=X.columns)
X_test_with_cluster['cluster'] = cluster_labels_test

# 7. RandomForestClassifier로 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train_with_cluster, y_train)

# 8. 예측 및 평가
y_pred = model.predict(X_test_with_cluster)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# print(f"F1 Score (with clustering feature): {f1:.4f}")

# print('train_accuracy: ', lr_clf.score(X_train_scaled, y_train))
# print('train_accuracy: ', lr_clf.score(X_train_resampled, y_train_resampled))
# print('val_accuracy: ', lr_clf.score(X_val_scaled, y_val))

print(classification_report(y_test, y_pred))
print('AUC-ROC:', roc_auc_score(y_test, y_pred))

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')

