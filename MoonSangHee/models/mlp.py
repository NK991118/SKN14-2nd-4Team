import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd

class DNN(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.linear1 = nn.Linear(Cin, 32)
        self.linear2 = nn.Linear(32, 16)
        # self.linear3 = nn.Linear(32, 16)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        # out = self.linear3(out)
        # out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out

data = pd.read_csv('./data/train.csv')
# X = torch.tensor(data.drop('Churn', axis=1).to_numpy(), dtype=torch.float32) #(3200, 12)
X = data.drop('Churn', axis=1)
y = torch.tensor(data['Churn'].to_numpy(), dtype=torch.float32).unsqueeze(dim=1) #(3200, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)

# scaler = StandardScaler()
scaler = RobustScaler()
X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_val_scaled =  torch.tensor(scaler.transform(X_val), dtype=torch.float32)

model = DNN(Cin = X_train_scaled.size(1))
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay=1e-4)

model.train()
best_f1score=  0.
for epoch in range(1000):

  optimizer.zero_grad() 
  y_pred = model(X_train_scaled).type(torch.float32)
  loss = criterion(y_pred, y_train)
  loss.backward() 
  optimizer.step() 

  if (epoch + 1) % 10 == 0:
    print(f'Epoch: {epoch + 1} Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():

      y_val_pred = model(X_val_scaled)
      loss = criterion(y_val_pred, y_val)
     
      y_val_prob = F.sigmoid(y_val_pred)  
      y_val_pred_label = (y_val_prob >= 0.5).float() 

      y_true = y_val.numpy()
      y_pred_labels = y_val_pred_label.numpy()
      
      acc = accuracy_score(y_true, y_pred_labels)
      precision = precision_score(y_true, y_pred_labels)
      recall = recall_score(y_true, y_pred_labels)
      f1 = f1_score(y_true, y_pred_labels)

      print(f'EVAL loss: {loss:.4f} Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

      if best_f1score < f1:
        torch.save(model.state_dict(), './best_model.pth')

        with open('scaler_mlp.pkl', 'wb') as f:
          pickle.dump(scaler, f)
          
        print(f'====Best model saved at {epoch+1} with F1 Score: {f1:.4f}')
        best_f1score = f1



