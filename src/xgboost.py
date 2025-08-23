import pandas as pd
import numpy as np
import shap
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, roc_curve, auc, log_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def preprocess_feature_set(data):
    X = data.drop(columns=['Exacerbation.Outcome', 'subject_id'])
    y = data['Exacerbation.Outcome']
    return X, y

    
# Precompute SHAP Features for All Datasets (Before Cross-Validation)
all_shap_features_dict = {}
shap_feature_dfs = []

for dataset_id, feature_set in enumerate(data_list, 1):
    print(f"\n🔹 Computing SHAP Features for Dataset {dataset_id}")

    X = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
    y = feature_set['Exacerbation.Outcome']

    # Train Logistic Regression for SHAP Analysis
    logreg = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', 
                                C=100, max_iter=10000, random_state=42)
    logreg.fit(X, y)

    # Compute SHAP values
    explainer = shap.Explainer(logreg, X)
    shap_values = explainer(X)

    feature_importance_abs = np.abs(shap_values.values).mean(axis=0)
    feature_importance_signed = shap_values.values.mean(axis=0)
    logreg_coefs = logreg.coef_[0]

    # Select the top 200 SHAP features
    top_features = X.columns[np.argsort(feature_importance_abs)[-200:]].tolist()
    
    # Store selected features for this dataset
    all_shap_features_dict[dataset_id] = top_features

    # Save SHAP feature importances & LogReg coefficients
    top_200_features_df = pd.DataFrame({
        "Dataset_ID": dataset_id,
        "Feature": top_features,
        "SHAP_Abs": feature_importance_abs[np.argsort(feature_importance_abs)[-200:]],
        "SHAP_Signed": feature_importance_signed[np.argsort(feature_importance_abs)[-200:]],
        "LogReg_Coef": logreg_coefs[np.argsort(feature_importance_abs)[-200:]]
    })

    shap_feature_dfs.append(top_200_features_df)

# Save Precomputed SHAP Features
shap_features_df = pd.concat(shap_feature_dfs, ignore_index=True)
# shap_features_df.to_csv(r"C:\DATA\bayesian\data\interim\Preselected_SHAP_Features_200_XGB_Feb25.csv", index=False)
# print("SHAP Features Precomputed and Saved!")


# Train and evaluate XGBoost model
def train_xgboost(X, y, top_features, best_params):
    X_top = X[top_features]
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)
    model = XGBClassifier(**best_params, tree_method='hist', device='cuda', random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_test, y_pred_test)
    train_loss = log_loss(y_train, y_pred_proba_train)
    val_loss = log_loss(y_test, y_pred_proba_test)
    return model, train_acc, val_acc, train_loss, val_loss



# Hyperparameter tuning using Optuna
def optimize_xgboost(X, y):
    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
            'max_depth': trial.suggest_int('max_depth', 1, 4),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.3, 0.7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'lambda': trial.suggest_float('lambda', 0.1, 10, log=True),
            'alpha': trial.suggest_float('alpha', 0.1, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5)
        }
        # Optionally add scale_pos_weight if imbalanced
        param['scale_pos_weight'] = sum(y==0) / sum(y==1)
              
        
        xgb_clf = XGBClassifier(**param)
        xgb_clf.fit(X, y)
        preds = xgb_clf.predict(X)
        return accuracy_score(y, preds)




# # Hyperparameter tuning using Optuna
# def optimize_xgboost(X, y):
#     def objective(trial):
#         param = {
#             'objective': 'binary:logistic',
#             'tree_method': 'hist',
#             'device': 'cuda',
#             'random_state': 42,
#             'max_depth': trial.suggest_int('max_depth', 1, 4),
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#             'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'lambda': trial.suggest_float('lambda', 1, 25),
#             'alpha': trial.suggest_float('alpha', 0.1, 25),
#             'gamma': trial.suggest_float('gamma', 0, 10)
#         }
#         xgb_clf = XGBClassifier(**param)
#         xgb_clf.fit(X, y)
#         preds = xgb_clf.predict(X)
#         return accuracy_score(y, preds)


    

# # Hyperparameter tuning using Optuna
# def optimize_xgboost(X, y):
#     def objective(trial):
#         param = {
#             'objective': 'binary:logistic',
#             'tree_method': 'hist',
#             'device': 'cuda',
#             'random_state': 42,
#             'max_depth': trial.suggest_int('max_depth', 1, 3),
#             'min_child_weight': trial.suggest_int('min_child_weight', 5, 10),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
#             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#             'subsample': trial.suggest_float('subsample', 0.4, 0.8),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
#            'lambda': trial.suggest_float('lambda', 0, 20),
#             'alpha': trial.suggest_float('alpha', 0.1, 10),
#             'gamma': trial.suggest_float('gamma', 0, 10)
#         }
#         xgb_clf = XGBClassifier(**param)
#         xgb_clf.fit(X, y)
#         preds = xgb_clf.predict(X)
#         return accuracy_score(y, preds)
    
    

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

# K-Fold Cross-Validation Function
def k_fold_cross_validation(X, y, dataset_id, n_splits, top_features, best_params):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    val_precisions, val_recalls, val_roc_aucs = [], [], []
    roc_curve_data = []

    X_top = X[top_features]

    for fold_id, (train_index, test_index) in enumerate(kfold.split(X_top), start=1):
        print(f"Dataset {dataset_id}, Fold {fold_id}/{n_splits}")

        X_train, X_test = X_top.iloc[train_index], X_top.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = XGBClassifier(**best_params, tree_method='hist', device='cuda', random_state=42)
        model.fit(X_train, y_train)

        # Train metrics
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, y_pred_train)
        train_loss = log_loss(y_train, y_pred_proba_train)

        # Validation metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        val_acc = accuracy_score(y_test, y_pred)
        val_loss = log_loss(y_test, y_pred_proba)
        val_precision = precision_score(y_test, y_pred, average="binary")
        val_recall = recall_score(y_test, y_pred, average="binary")
        val_roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Store metrics
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_roc_aucs.append(val_roc_auc)

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_curve_data.append((fpr, tpr))

    # Print mean and std for all metrics
    print(f"Mean Train Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Mean Val Accuracy:   {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Mean Train Loss:     {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
    print(f"Mean Val Loss:       {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Mean Val Precision:  {np.mean(val_precisions):.4f} ± {np.std(val_precisions):.4f}")
    print(f"Mean Val Recall:     {np.mean(val_recalls):.4f} ± {np.std(val_recalls):.4f}")
    print(f"Mean Val ROC-AUC:    {np.mean(val_roc_aucs):.4f} ± {np.std(val_roc_aucs):.4f}")

    # Return all metrics if you want to aggregate later
    return {
        "train_acc": train_accs,
        "val_acc": val_accs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_precision": val_precisions,
        "val_recall": val_recalls,
        "val_roc_auc": val_roc_aucs,
        "roc_curve_data": roc_curve_data
    }


all_results = []
roc_data = []

for dataset_id, data in enumerate(data_list, start=1):
    print(f"\nRunning K-Fold CV for Dataset {dataset_id} with {10} folds")
    X, y = preprocess_feature_set(data)
    top_features = all_shap_features_dict[dataset_id]
    best_params = optimize_xgboost(X[top_features], y)
    results = k_fold_cross_validation(X, y, dataset_id, n_splits=10, top_features=top_features, best_params=best_params)
    # You can aggregate or print more here if needed

    # Compute mean metrics across folds
    mean_accuracy = np.mean(results["val_acc"])
    mean_roc_auc = np.mean(results["val_roc_auc"])
    mean_precision = np.mean(results["val_precision"])
    mean_recall = np.mean(results["val_recall"])
    mean_train_acc = np.mean(results["train_acc"])
    mean_val_acc = np.mean(results["val_acc"])
    mean_train_loss = np.mean(results["train_loss"])
    mean_val_loss = np.mean(results["val_loss"])

    # Print results
    print(f"\n### Results for Dataset {dataset_id} ###")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean Train Accuracy: {mean_train_acc:.4f} ± {np.std(results['train_acc']):.4f}")
    print(f"Mean Val Accuracy:   {mean_val_acc:.4f} ± {np.std(results['val_acc']):.4f}")
    print(f"Mean Train Loss:     {mean_train_loss:.4f} ± {np.std(results['train_loss']):.4f}")
    print(f"Mean Val Loss:       {mean_val_loss:.4f} ± {np.std(results['val_loss']):.4f}")

    # Store results
    all_results.append([dataset_id, mean_accuracy, mean_roc_auc, mean_precision, mean_recall, mean_train_acc, mean_val_acc, mean_train_loss, mean_val_loss])
    
    # Store ROC curve data
    roc_data.extend(results["roc_curve_data"])

# Compute mean ROC curve across datasets
mean_fpr_ClinGenDivXGB_200 = np.linspace(0, 1, 100)
mean_tpr_ClinGenDivXGB_200 = np.mean([np.interp(mean_fpr_ClinGenDivXGB_200, fpr, tpr) for fpr, tpr in roc_data], axis=0)
std_tpr_ClinGenDivXGB_200 = np.std([np.interp(mean_fpr_ClinGenDivXGB_200, fpr, tpr) for fpr, tpr in roc_data], axis=0)
final_auc_ClinGenDivXGB_200 = auc(mean_fpr_ClinGenDivXGB_200, mean_tpr_ClinGenDivXGB_200)

print(f"\nFinal Mean ROC-AUC across datasets: {final_auc_ClinGenDivXGB_200:.4f}")

# Convert results to DataFrame
df_results = pd.DataFrame(all_results, columns=["Dataset", "Mean Accuracy", "Mean ROC AUC", "Mean Precision", "Mean Recall", "Mean Train Accuracy", "Mean Val Accuracy", "Mean Train Loss", "Mean Val Loss"])
print("\nFinal Results Across Datasets:")
print(df_results)

# Print final mean performance
print(f"\nOverall Mean Accuracy: {df_results['Mean Accuracy'].mean():.4f}")
print(f"Overall Mean ROC-AUC: {df_results['Mean ROC AUC'].mean():.4f}")
print(f"Overall Mean Precision: {df_results['Mean Precision'].mean():.4f}")
print(f"Overall Mean Recall: {df_results['Mean Recall'].mean():.4f}")

# Save Results
# df_results.to_csv(r"C:\XGBoost_Results.csv", index=False)

# Save Results
# df_results.to_csv(r"C:\XGBoost_Results.csv", index=False)


# Calculate overall metrics with std
print("\nOverall Metrics:")
print(f"Mean Accuracy: {df_results['Mean Accuracy'].mean():.4f} ± {df_results['Mean Accuracy'].std():.4f}")
print(f"Mean ROC-AUC Score: {df_results['Mean ROC AUC'].mean():.4f} ± {df_results['Mean ROC AUC'].std():.4f}")
print(f"Mean Precision: {df_results['Mean Precision'].mean():.4f} ± {df_results['Mean Precision'].std():.4f}")
print(f"Mean Recall: {df_results['Mean Recall'].mean():.4f} ± {df_results['Mean Recall'].std():.4f}")

# Calculate metrics per dataset
print("\nPer Dataset Performance:")
for dataset_id in range(1, 6):
    dataset_results = df_results[df_results['Dataset'] == dataset_id]
    
    print(f"\nDataset {dataset_id} Performance:")
    print(f"Accuracy: {dataset_results['Mean Accuracy'].iloc[0]:.4f}")
    print(f"ROC-AUC Score: {dataset_results['Mean ROC AUC'].iloc[0]:.4f}")
    print(f"Precision: {dataset_results['Mean Precision'].iloc[0]:.4f}")
    print(f"Recall: {dataset_results['Mean Recall'].iloc[0]:.4f}")

all_results = np.array(all_results)
mean_train_accs = all_results[:, 5].astype(float)  # Mean Train Accuracy per dataset
mean_val_accs = all_results[:, 6].astype(float)    # Mean Val Accuracy per dataset
mean_train_losses = all_results[:, 7].astype(float)  # Mean Train Loss per dataset
mean_val_losses = all_results[:, 8].astype(float)    # Mean Val Loss per dataset

print("\n=== Overall Mean/Std Across All Datasets ===")
print(f"Mean Train Accuracy: {np.mean(mean_train_accs):.4f} ± {np.std(mean_train_accs):.4f}")
print(f"Mean Val Accuracy:   {np.mean(mean_val_accs):.4f} ± {np.std(mean_val_accs):.4f}")
print(f"Mean Train Loss:     {np.mean(mean_train_losses):.4f} ± {np.std(mean_train_losses):.4f}")
print(f"Mean Val Loss:       {np.mean(mean_val_losses):.4f} ± {np.std(mean_val_losses):.4f}")




# for ROC curve 
#======================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# roc_data = list of (fpr, tpr) tuples for each fold

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

for fpr, tpr in roc_data:
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(auc(fpr, tpr))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

plt.figure()
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve - XGB (K-Fold CV)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


np.save('xgb_mean_fpr_Gene_200.npy', mean_fpr)
np.save('xgb_mean_tpr_Gene_200.npy', mean_tpr)


mean_fpr = np.load('xgb_mean_fpr_Gene_200.npy')
mean_tpr = np.load('xgb_mean_tpr_Gene_200.npy')

plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve - XGB')
plt.legend()
plt.show()


