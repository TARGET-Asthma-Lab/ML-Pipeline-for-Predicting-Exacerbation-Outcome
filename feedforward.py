# hypertuning for 60 features or 200 features

'''
Uncomment the lines corresponding to the data combination you want to run, and leave the others commented.  
After execution, collect the resulting metrics and figures, and save them to your preferred location.  
Then, proceed to the next data combination and run the model again.  

Example:  
Uncomment # Gene Only  
data_list = [pd.read_csv(rf'gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv') for i in range(1, 6)]  
and keep others (Clinical, ClinGen, RawDiv, etc.) commented.  

Run the model for Gene data only.  

If you need a combination of the data with alpha and beta diversity or raw diversity, then:  
1. Uncomment the data combination you want to run.  
2. Uncomment the `alpha_div_list` and `beta_div_list` lines if you want to test your data with alpha-beta diversities.  
3. Uncomment the merging and processing block for Raw Diversity if you want to test your data with Raw Diversity.  

Example: Uncomment # Gene Only  
data_list = [pd.read_csv(rf'gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv') for i in range(1, 6)]  

alpha_div_list = [pd.read_csv(rf'alpha_div_imputed_pmm{i}_Jan30.csv') for i in range(1, 6)]  
beta_div_list = [pd.read_csv(rf'beta_div_imputed_pmm{i}_Jan30.csv') for i in range(1, 6)]  

Run the model for Gene data + alpha-beta diversities.  

To run the model with either n=60 or n=200 Preselected SHAP features, choose the corresponding code block for either n=60 or n=200.
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import shap
import copy
import optuna
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.decomposition import PCA
from torchsummary import summary


# Set seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Gene Only
data_list  = [pd.read_csv(rf'gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv') for i in range(1, 6)]

# # Clinical
# data_list  = [pd.read_csv(rf'clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv') for i in range(1, 6)]

# # # ClinGen
# data_list  = [pd.read_csv(rf'gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv') for i in range(1, 6)]

# ## Raw Div
# raw_div = pd.read_csv(r"otutab_transp_div_imputed_fastFeb13.csv")

## Exacerbation.Outcom only
# data_list  = [pd.read_csv(rf'clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_ExacerOut.csv') for i in range(1, 6)]


for i, df in enumerate(data_list, 1):
    print(f"\nDataset {i} Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Exacerbation ratio: {df['Exacerbation.Outcome'].mean():.3f}")
    print(f"Number of features: {df.shape[1]-2}")  # -2 for Exacerbation.Outcome and subject_id

# # Load alpha and beta diversity datasets
# alpha_div_list = [pd.read_csv(rf'alpha_div_imputed_pmm{i}_Jan30.csv') for i in range(1, 6)]

# beta_div_list = [pd.read_csv(rf'beta_div_imputed_pmm{i}_Jan30.csv') for i in range(1, 6)]


# #Merge each dataset with its corresponding alpha/beta diversity
# merged_data_list = []
# for i in range(5):
#     merged_df = data_list[i].merge(alpha_div_list[i], on='subject_id', how='inner')
#     merged_df = merged_df.merge(beta_div_list[i], on='subject_id', how='inner')
#     merged_data_list.append(merged_df)



# ## Merging and processing Raw Div 

# # ===========================
# # Standardize & Apply PCA to Raw Diversity Data
# # ===========================

# scaler = StandardScaler()
# diversity_features = [col for col in raw_div.columns if col != 'subject_id']
# raw_div[diversity_features] = scaler.fit_transform(raw_div[diversity_features])

# pca = PCA(n_components=0.95)  # Retain 95% variance
# pca_transformed = pca.fit_transform(raw_div[diversity_features])
# pca_columns = [f'PCA_{i+1}' for i in range(pca_transformed.shape[1])]

# pca_df = pd.DataFrame(pca_transformed, columns=pca_columns)
# pca_df['subject_id'] = raw_div['subject_id']

# print(f"Raw diversity PCA reduced from {len(diversity_features)} to {pca_transformed.shape[1]} components.")

# # ===========================
# # Merge Data (With PCA Features)
# # ===========================

# ## Gene + Clin + Div + RawDiv merge

# merged_data_list = []
# for i in range(5):
#     merged_df = data_list[i].merge(alpha_div_list[i], on='subject_id', how='inner')
#     merged_df = merged_df.merge(beta_div_list[i], on='subject_id', how='inner')
#     merged_df = merged_df.merge(pca_df, on='subject_id', how='inner')  # Use PCA-transformed diversity data
#     merged_data_list.append(merged_df)




# Precompute SHAP Features for All Datasets (Before Cross-Validation), n=60

all_shap_features_dict = {}
shap_feature_dfs = []

for dataset_id, feature_set in enumerate(data_list, 1):
    print(f"\n🔹 Computing SHAP Features for Dataset {dataset_id}")

    X = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
    y = feature_set['Exacerbation.Outcome']

    logreg = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', C=100, max_iter=10000, random_state=42)
    logreg.fit(X, y)

    explainer = shap.Explainer(logreg, X)
    shap_values = explainer(X)

    feature_importance_abs = np.abs(shap_values.values).mean(axis=0)
    feature_importance_signed = shap_values.values.mean(axis=0)
    logreg_coefs = logreg.coef_[0]

    # Select the top 60 features
    top_features = X.columns[np.argsort(feature_importance_abs)[-60:]].tolist()
    
    all_shap_features_dict[dataset_id] = top_features

    # Save SHAP feature importances & LogReg coefficients
    top_60_features_df = pd.DataFrame({
        "Dataset_ID": dataset_id,
        "Feature": top_features,
        "SHAP_Abs": feature_importance_abs[np.argsort(feature_importance_abs)[-60:]],
        "SHAP_Signed": feature_importance_signed[np.argsort(feature_importance_abs)[-60:]],
        "LogReg_Coef": logreg_coefs[np.argsort(feature_importance_abs)[-60:]]
    })

    shap_feature_dfs.append(top_60_features_df)

# Save Precomputed SHAP Features
shap_features_df = pd.concat(shap_feature_dfs, ignore_index=True)
shap_features_df.to_csv(r"Preselected_SHAP_FFnn_GenClinDiv_May27_C100_num_features_60.csv", index=False)
print("SHAP Features Precomputed and Saved!")


# # Precompute SHAP Features for All Datasets (Before Cross-Validation), n=200

# all_shap_features_dict = {}
# shap_feature_dfs = []

# for dataset_id, feature_set in enumerate(data_list, 1):
#     print(f"\n🔹 Computing SHAP Features for Dataset {dataset_id}")

#     X = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
#     y = feature_set['Exacerbation.Outcome']

#     logreg = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', C=100, max_iter=10000, random_state=42)
#     logreg.fit(X, y)

#     explainer = shap.Explainer(logreg, X)
#     shap_values = explainer(X)

#     feature_importance_abs = np.abs(shap_values.values).mean(axis=0)
#     feature_importance_signed = shap_values.values.mean(axis=0)
#     logreg_coefs = logreg.coef_[0]

#     # Select the top 60 features
#     top_features = X.columns[np.argsort(feature_importance_abs)[-200:]].tolist()
    
#     all_shap_features_dict[dataset_id] = top_features

#     # Save SHAP feature importances & LogReg coefficients
#     top_200_features_df = pd.DataFrame({
#         "Dataset_ID": dataset_id,
#         "Feature": top_features,
#         "SHAP_Abs": feature_importance_abs[np.argsort(feature_importance_abs)[-200:]],
#         "SHAP_Signed": feature_importance_signed[np.argsort(feature_importance_abs)[-200:]],
#         "LogReg_Coef": logreg_coefs[np.argsort(feature_importance_abs)[-200:]]
#     })

#     shap_feature_dfs.append(top_200_features_df)

# # Save Precomputed SHAP Features
# shap_features_df = pd.concat(shap_feature_dfs, ignore_index=True)
# shap_features_df.to_csv(r"Preselected_SHAP_FFnn_GenClinDiv_May27_C100_num_features_200.csv", index=False)
# print("SHAP Features Precomputed and Saved!")


###########################################
# 3. Define the Feedforward Neural Network
###########################################
# for 60 features

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    
###########################################
# 4. Training and K-Fold Cross-Validation
###########################################

# Store performance metrics for all datasets
all_results = []
roc_data = []  # Store ROC curve data
dataset_histories = {i: {'train_losses': [], 'val_losses': [], 'train_accuracies': [], 'val_accuracies': []} 
                    for i in range(1, 6)}

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):  # Increased patience, smaller min_delta
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# =====================
# Optuna Hyperparameter Tuning Block (for 60 features)
# =====================
import optuna

def optuna_objective(trial):
    # Sample hyperparameters
    n_hidden1 = trial.suggest_categorical('n_hidden1', [32, 64, 128])
    n_hidden2 = trial.suggest_categorical('n_hidden2', [16, 32, 64])
   
    dropout_rate = trial.suggest_float('dropout', 0.3, 0.7)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    # Use first dataset for speed
    feature_set = data_list[0]
    top_features = all_shap_features_dict[1]
    X = feature_set[top_features]
    y = feature_set['Exacerbation.Outcome']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Dynamic model class
    class TunedFFNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, n_hidden1)
            self.bn1 = nn.BatchNorm1d(n_hidden1)
            self.fc2 = nn.Linear(n_hidden1, n_hidden2)
            self.bn2 = nn.BatchNorm1d(n_hidden2)
            self.fc3 = nn.Linear(n_hidden2, 1)
            self.dropout = nn.Dropout(dropout_rate)
            self._initialize_weights()
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    model = TunedFFNN(input_dim=X_train_scaled.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Short training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1).to(device)
        noise = torch.randn_like(X_train_tensor) * 0.01
        output = model(X_train_tensor + noise)
        train_loss = criterion(output, y_train_tensor)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation loss
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float).to(device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float).view(-1, 1).to(device)
        val_output = model(X_val_tensor)
        val_loss = criterion(val_output, y_val_tensor)
        val_loss = val_loss.item()
    return val_loss

# Uncomment to run tuning (will take a few minutes)
# study = optuna.create_study(direction='minimize')
# study.optimize(optuna_objective, n_trials=30)
# print('Best trial:', study.best_trial.params)

# =====================
# End Optuna Block
# =====================

# =====================
# Comment out main training loop for now (run Optuna first)
# =====================
# for dataset_idx, feature_set in enumerate(data_list, start=1):
#     ... (rest of your main loop)
# =====================

# =====================
# Main training loop with best hyperparameters from Optuna
# =====================
# Store performance metrics for all datasets
all_results = []
roc_data = []  # Store ROC curve data
dataset_histories = {i: {'train_losses': [], 'val_losses': [], 'train_accuracies': [], 'val_accuracies': []} 
                    for i in range(1, 6)}

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):  # Increased patience, smaller min_delta
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
# bootstrap for CI calculation
def bootstrap_auc(y_true, y_pred, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip if only one class in the sample
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    mean = np.mean(aucs)
    return mean, lower, upper

for dataset_idx, feature_set in enumerate(data_list, start=1):
    print(f"\n🔹 Processing Dataset {dataset_idx}")

    # Store accumulated losses and accuracies for this dataset
    accumulated_train_losses = np.zeros(500)  # Change back to 500 epochs
    accumulated_val_losses = np.zeros(500)
    accumulated_train_accuracies = np.zeros(500)
    accumulated_val_accuracies = np.zeros(500)
    fold_counts = np.zeros(500)  # Track how many folds contributed to each epoch

    for fold in range(10):
        print(f"\nFold {fold + 1}/10")

        # Load Precomputed SHAP Features
        top_features = all_shap_features_dict[dataset_idx]
        X = feature_set[top_features]
        y = feature_set['Exacerbation.Outcome']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FeedforwardNN(input_dim=X_train_scaled.shape[1], dropout_rate=0.7).to(device)

        # Before training loop
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(model.parameters(), lr=0.007, weight_decay=0.01) # was 0.05 (for higher n=200?), because 0.01 is perfect for n=60
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Initialize lists for this fold
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=20)

        for epoch in range(500):  # Change back to 500 epochs
            model.train()
            optimizer.zero_grad()
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float).to(device)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1).to(device)
            
            noise = torch.randn_like(X_train_tensor) * 0.05
            output = model(X_train_tensor + noise)
            train_loss = criterion(output, y_train_tensor)
            train_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Compute training metrics
            with torch.no_grad():
                train_probs = torch.sigmoid(output)
                train_preds = (train_probs > 0.5).float()
                train_acc = accuracy_score(y_train.values, train_preds.cpu().numpy())
            
            # Validation step
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to(device)
                y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1).to(device)
                val_output = model(X_test_tensor)
                val_loss = criterion(val_output, y_test_tensor)
                val_acc = accuracy_score(y_test, (torch.sigmoid(val_output) > 0.5).float().cpu().numpy())

            # Store metrics
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # Early stopping check
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(early_stopping.best_model)
                break

        # Proper accumulation of metrics
        n_epochs = len(train_losses)
        accumulated_train_losses[:n_epochs] += np.array(train_losses)
        accumulated_val_losses[:n_epochs] += np.array(val_losses)
        accumulated_train_accuracies[:n_epochs] += np.array(train_accuracies)
        accumulated_val_accuracies[:n_epochs] += np.array(val_accuracies)
        fold_counts[:n_epochs] += 1

    # Proper averaging of metrics
    mask = fold_counts > 0
    dataset_histories[dataset_idx]['train_losses'] = accumulated_train_losses[mask] / fold_counts[mask]
    dataset_histories[dataset_idx]['val_losses'] = accumulated_val_losses[mask] / fold_counts[mask]
    dataset_histories[dataset_idx]['train_accuracies'] = accumulated_train_accuracies[mask] / fold_counts[mask]
    dataset_histories[dataset_idx]['val_accuracies'] = accumulated_val_accuracies[mask] / fold_counts[mask]

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_scaled, dtype=torch.float).to(device))
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()

    test_accuracy = accuracy_score(y_test, predictions.cpu().numpy())
    roc_auc = roc_auc_score(y_test, probabilities.cpu().numpy())
    precision = precision_score(y_test, predictions.cpu().numpy(), average="binary")
    recall = recall_score(y_test, predictions.cpu().numpy(), average="binary")

    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    probs_np = probabilities.cpu().numpy().flatten()
    mean_auc, lower_auc, upper_auc = bootstrap_auc(y_test_np, probs_np, n_bootstrap=1000)

    ci_width = (upper_auc - lower_auc) / 2
    print(f"Fold {fold+1} AUC: {roc_auc:.4f} (bootstrap: {mean_auc:.4f} ± {ci_width:.4f})")

    fold_metrics = [dataset_idx, fold + 1, test_accuracy, roc_auc, precision, recall, mean_auc, lower_auc, upper_auc]
    all_results.append(fold_metrics)
    fpr, tpr, _ = roc_curve(y_test, probabilities.cpu().numpy())
    roc_data.append((fpr, tpr))

    # Plot training history per dataset
    actual_epochs = len(dataset_histories[dataset_idx]['train_losses'])  # Get actual number of epochs
    epochs = range(1, actual_epochs + 1)  # Use actual epochs instead of fixed 500
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, dataset_histories[dataset_idx]['train_losses'], label='Training Loss', color='blue')
    plt.plot(epochs, dataset_histories[dataset_idx]['val_losses'], label='Validation Loss', color='red')
    plt.title(f'Average Training vs Validation Loss (Dataset {dataset_idx})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, dataset_histories[dataset_idx]['train_accuracies'], label='Training Accuracy', color='blue')
    plt.plot(epochs, dataset_histories[dataset_idx]['val_accuracies'], label='Validation Accuracy', color='red')
    plt.title(f'Average Training vs Validation Accuracy (Dataset {dataset_idx})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

# Calculate and plot overall average across all datasets
plt.figure(figsize=(15, 5))

# Get minimum length across all histories to align the data
min_length = min(len(hist['train_losses']) for hist in dataset_histories.values())

# Truncate all histories to the minimum length
for dataset_idx in dataset_histories:
    for metric in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']:
        dataset_histories[dataset_idx][metric] = dataset_histories[dataset_idx][metric][:min_length]

# Calculate overall means with aligned lengths
overall_train_losses = np.mean([hist['train_losses'][:min_length] for hist in dataset_histories.values()], axis=0)
overall_val_losses = np.mean([hist['val_losses'][:min_length] for hist in dataset_histories.values()], axis=0)
overall_train_accuracies = np.mean([hist['train_accuracies'][:min_length] for hist in dataset_histories.values()], axis=0)
overall_val_accuracies = np.mean([hist['val_accuracies'][:min_length] for hist in dataset_histories.values()], axis=0)

# Calculate standard deviations with aligned lengths
train_losses_std = np.std([hist['train_losses'][:min_length] for hist in dataset_histories.values()], axis=0)
val_losses_std = np.std([hist['val_losses'][:min_length] for hist in dataset_histories.values()], axis=0)
train_accuracies_std = np.std([hist['train_accuracies'][:min_length] for hist in dataset_histories.values()], axis=0)
val_accuracies_std = np.std([hist['val_accuracies'][:min_length] for hist in dataset_histories.values()], axis=0)

epochs = range(1, min_length + 1)

# Plot Overall Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, overall_train_losses, label='Average Training Loss', color='blue')
plt.fill_between(epochs, overall_train_losses - train_losses_std, 
                overall_train_losses + train_losses_std, alpha=0.2, color='blue')
plt.plot(epochs, overall_val_losses, label='Average Validation Loss', color='red')
plt.fill_between(epochs, overall_val_losses - val_losses_std,
                overall_val_losses + val_losses_std, alpha=0.2, color='red')
plt.title('Overall Average Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Overall Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, overall_train_accuracies, label='Average Training Accuracy', color='blue')
plt.fill_between(epochs, overall_train_accuracies - train_accuracies_std,
                overall_train_accuracies + train_accuracies_std, alpha=0.2, color='blue')
plt.plot(epochs, overall_val_accuracies, label='Average Validation Accuracy', color='red')
plt.fill_between(epochs, overall_val_accuracies - val_accuracies_std,
                overall_val_accuracies + val_accuracies_std, alpha=0.2, color='red')
plt.title('Overall Average Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Print final performance metrics
df_results = pd.DataFrame(all_results, columns=["Dataset", "Fold", "Test Accuracy", "ROC-AUC", "Precision", "Recall", "Mean AUC", "Lower AUC", "Upper AUC"])
print("\n🔹 Final Model Performance Across All Datasets and Folds:")
print(f"Mean Test Accuracy: {df_results['Test Accuracy'].mean():.4f} ± {df_results['Test Accuracy'].std():.4f}")
print(f"Mean ROC-AUC Score: {df_results['ROC-AUC'].mean():.4f} ± {df_results['ROC-AUC'].std():.4f}")
print(f"Mean Precision: {df_results['Precision'].mean():.4f} ± {df_results['Precision'].std():.4f}")
print(f"Mean Recall: {df_results['Recall'].mean():.4f} ± {df_results['Recall'].std():.4f}")

print(f"\nFinal Average Training Loss (Epoch {min_length}): {overall_train_losses[-1]:.4f} ± {train_losses_std[-1]:.4f}")
print(f"Final Average Validation Loss (Epoch {min_length}): {overall_val_losses[-1]:.4f} ± {val_losses_std[-1]:.4f}")
print(f"Final Average Training Accuracy (Epoch {min_length}): {overall_train_accuracies[-1]:.4f} ± {train_accuracies_std[-1]:.4f}")
print(f"Final Average Validation Accuracy (Epoch {min_length}): {overall_val_accuracies[-1]:.4f} ± {val_accuracies_std[-1]:.4f}")



# Print per-dataset performance
for dataset_idx in range(1, 6):
    dataset_results = df_results[df_results['Dataset'] == dataset_idx]
    print(f"\nDataset {dataset_idx} Performance:")
    print(f"Test Accuracy: {dataset_results['Test Accuracy'].mean():.4f} ± {dataset_results['Test Accuracy'].std():.4f}")
    print(f"ROC-AUC Score: {dataset_results['ROC-AUC'].mean():.4f} ± {dataset_results['ROC-AUC'].std():.4f}")
    

# Calculate mean and mean half-width of bootstrap AUC CI
mean_auc = df_results['Mean AUC'].mean()
mean_ci_width = ((df_results['Upper AUC'] - df_results['Lower AUC']) / 2).mean()

print(f"Mean Bootstrapped AUC: {mean_auc:.4f} ± {mean_ci_width:.4f}")



# ==================================
# Mean ROC Curve Calculation and Plot
# ==================================
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

# roc_data contains (fpr, tpr) for each fold from all datasets
for fpr, tpr in roc_data:
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(auc(fpr, tpr))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc_from_roc = np.mean(aucs)
std_auc_from_roc = np.std(aucs)

plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=f'Mean ROC (AUC = {mean_auc_from_roc:.2f} ± {std_auc_from_roc:.2f})',
         lw=2, alpha=.8)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red',
         label='Chance', alpha=.8)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve - FFNN (Cross-Validated)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Save the mean FPR and TPR arrays
np.save('ffnn_mean_fpr_Gene_200.npy', mean_fpr)
np.save('ffnn_mean_tpr_Gene_200.npy', mean_tpr)
# print("\nSaved mean ROC curve data to 'ffnn_mean_fpr_ClinGenDiv_200.npy' and 'ffnn_mean_tpr_ClinGenDiv_200.npy'.")


# # Example usage: print model summary for FeedforwardNN
# if __name__ == "__main__":
#     input_dim =60  # Change as needed for your data
#     model = FeedforwardNN(input_dim=input_dim, dropout_rate=0.7)
#     summary(model, input_size=(1, input_dim))
    
# summary(model, input_size=(input_dim,))


# Summary parameteres
from torchinfo import summary

if __name__ == "__main__":
    input_dim = 60
    model = FeedforwardNN(input_dim=input_dim, dropout_rate=0.7)
    model.eval()
summary(model, input_size=(1, input_dim))
  
