# import torch, pickle
import pickle
# import torch.nn as nn
# import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd
# from mlp import DNN

test_cfg = {
    'file_type': 'pkl', #pth, pkl
    'pth_dir': './models/best_model.pth',
    'pkl_dir': './models/best_models/SVC.pkl',#LogisticRegression, XGBClassifier, LGBMClassifier, MLPClassifier, DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier, SVC
    'scaler_dir': './models/best_models/SVC_scaler.pkl.'
}


def test_pth(X_test, y_test, model_dir):

    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    model = DNN(Cin=X_test.size(1))

    model.load_state_dict(torch.load(model_dir))

    model.eval()

    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_prob = F.sigmoid(y_test_pred) 
        y_test_pred_label = (y_test_prob >= 0.5).float()

    y_true = y_test.numpy()
    y_pred_labels = y_test_pred_label.numpy()

    acc = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)

    return acc, precision, recall, f1, model.__class__.__name__

    

def test_pkl(X_test, y_test, model_dir):
    
    with open(model_dir, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, precision, recall, f1, model.__class__.__name__


def main():
    test_data = pd.read_csv('./data/test.csv')
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn'].to_numpy()
    
    with open(test_cfg['scaler_dir'], 'rb') as f:
        scaler = pickle.load(f)

    X_test_scaled = scaler.transform(X_test)

    if test_cfg['file_type'] == 'pth':
        acc, precision, recall, f1, model_name = test_pth(X_test_scaled, y_test, test_cfg['pth_dir'])

    elif test_cfg['file_type'] == 'pkl':
        acc, precision, recall, f1, model_name = test_pkl(X_test_scaled, y_test, test_cfg['pkl_dir'])

    else:
        raise ValueError(f'Unsupported file_type: {test_cfg['file_type']}')

    print(f'{model_name} Test Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    with open('./models/res.txt', '+a') as a:
        a.write(f'{model_name} Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f} \n')

if __name__ == '__main__':
    main()