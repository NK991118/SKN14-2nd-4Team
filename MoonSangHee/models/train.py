import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle

model_cfg = {
    'LogisticRegression': {'C': 2.0594494295802446},
    'XGBClassifier': {'max_depth': 3, 'n_estimators': 100, 'scale_pos_weight': 1.38},
    'LGBMClassifier': {'verbosity' : 1},
    'RandomForestClassifier': {'max_depth': 18, 'n_estimators': 131},
    'SVC': {'C': 2.4512552224777417, 'kernel': 'linear'},
    'KNeighborsClassifier': {'n_neighbors': 19},
    'MLPC': {'activation': 'relu','alpha': 0.0001, 'hidden_layer_sizes': (128,),'learning_rate': 'constant','learning_rate_init': 0.001,'solver': 'adam'},
    'VotingClassifier':{},
    'save_model' : True
}

def model_train_val(model, model_cfg, X_train, y_train, X_val, y_val):
    
    model_name = model.__class__.__name__
    if model_name in model_cfg:
        params = model_cfg[model_name]
        model.set_params(**params)
        print(f'Set parameters for {model_name}:{params}')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"model's name : {model.__class__.__name__}")

    print(classification_report(y_val, y_pred))
    print('val_accuracy: ', model.score(X_val, y_val))
    print('AUC-ROC:', roc_auc_score(y_val, y_pred))
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')

    return model



def main():

    df = pd.read_csv('./data/train.csv')

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled =  scaler.transform(X_val)

    # smote = SMOTE(random_state=42, sampling_strategy = 0.6)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    model = SVC() #LogisticRegression, XGBClassifier, LGBMClassifier, MLPClassifier, DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier, SVC

    model = model_train_val(model, model_cfg, X_train_scaled, y_train, X_val_scaled, y_val)

    if model_cfg['save_model'] == True:
        # print(model.__class__.__name__)
        dir = './models/' + model.__class__.__name__ + '.pkl'
        with open(dir, 'wb') as f:
            pickle.dump(model, f)

        scaler_dir = './models/' + model.__class__.__name__ + '_scaler.pkl'
        with open(scaler_dir, 'wb') as s:
          pickle.dump(scaler, s)
            


    

if __name__ == '__main__':
    main()