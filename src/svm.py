
import pandas as pd
import numpy as np
import shap
import optuna
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, hinge_loss
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import hinge_loss


# #  Load datasets
# data_list  = [pd.read_csv(rf'gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv') for i in range(1, 6)]


# Gene only
data_list  = [pd.read_csv(rf'gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv') for i in range(1, 6)]

# # Clinical
# data_list  = [pd.read_csv(rf'clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv') for i in range(1, 6)]


# #  For Diversity only 
# data_list  = [pd.read_csv(rf'clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_ExacerOut.csv') for i in range(1, 6)]


# raw_div = pd.read_csv(r"otutab_transp_div_imputed_fastFeb13.csv")



# # # Load alpha and beta diversity datasets
# alpha_div_list = [pd.read_csv(rf'alpha_div_imputed_pmm{i}_Jan30.csv') for i in range(1, 6)]
# beta_div_list = [pd.read_csv(rf'beta_div_imputed_pmm{i}_Jan30.csv') for i in range(1, 6)]



# # Merge each dataset with its corresponding alpha/beta diversity
# merged_data_list = []
# for i in range(5):
#     merged_df = data_list[i].merge(alpha_div_list[i], on='subject_id', how='inner')
#     merged_df = merged_df.merge(beta_div_list[i], on='subject_id', how='inner')
#     merged_data_list.append(merged_df)




# Precompute SHAP Features for All Datasets (Before Cross-Validation)
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
    top_features = X.columns[np.argsort(feature_importance_abs)[-200:]].tolist()
    
    all_shap_features_dict[dataset_id] = top_features

    # Save SHAP feature importances & LogReg coefficients
    top_60_features_df = pd.DataFrame({
        "Dataset_ID": dataset_id,
        "Feature": top_features,
        "SHAP_Abs": feature_importance_abs[np.argsort(feature_importance_abs)[-200:]],
        "SHAP_Signed": feature_importance_signed[np.argsort(feature_importance_abs)[-200:]],
        "LogReg_Coef": logreg_coefs[np.argsort(feature_importance_abs)[-200:]]
    })

    shap_feature_dfs.append(top_60_features_df)

# Save Precomputed SHAP Features
shap_features_df = pd.concat(shap_feature_dfs, ignore_index=True)
# shap_features_df.to_csv(r"C:\DATA\bayesian\data\interim\Preselected_SHAP_Features_SVM_Feb17.csv", index=False)
# print("SHAP Features Precomputed and Saved!")


# Train and evaluate SVM

def train_svm(X, y, top_features):
    X_top = X[top_features]
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.0001, 0.0005, 0.001],
        'kernel': ['linear']
    }
    grid_search = GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_svm = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")

    y_pred = best_svm.predict(X_test_scaled)
    y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Only extract coefficients if linear kernel
    if best_svm.kernel == 'linear':
        coef_dict = dict(zip(top_features, best_svm.coef_[0]))
    else:
        coef_dict = {}

    # Overfitting/underfitting diagnostics
    y_train_pred = best_svm.predict(X_train_scaled)
    y_train_proba = best_svm.decision_function(X_train_scaled)
    y_test_proba = best_svm.decision_function(X_test_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_hinge = hinge_loss(y_train, y_train_proba)
    val_hinge = hinge_loss(y_test, y_test_proba)

    return {
        'model': best_svm,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'coef_dict': coef_dict,
        'train_acc': train_acc,
        'val_acc': accuracy,
        'train_hinge': train_hinge,
        'val_hinge': val_hinge,
        'best_params': grid_search.best_params_
    }

###########################################
# K-Fold Cross-Validation
###########################################

def k_fold_cross_validation(X, y, dataset_id, n_splits=10):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    overall_results = {"accuracy": [], "roc_auc": [], "precision": [], "recall": [], "roc_curve_data": [], "svm_coefs": []}
    all_folds_results = []

    for fold_id, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        print(f"Dataset {dataset_id}, Fold {fold_id}")

        # Always select only the top features for both train and test
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        top_features = all_shap_features_dict[dataset_id]
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_top)
        X_test_scaled = scaler.transform(X_test_top)

        # Grid search and fit
        param_grid = {
            'C': [0.0001, 0.0005, 0.001],
            'kernel': ['linear']
        }
        grid = GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_params = grid.best_params_
        print(f"Dataset {dataset_id} Fold {fold_id} best params: {best_params}")

        best_svm = SVC(**best_params, probability=True, random_state=42)
        best_svm.fit(X_train_scaled, y_train)

        # Metrics
        y_pred = best_svm.predict(X_test_scaled)
        y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        overall_results["accuracy"].append(acc)
        overall_results["roc_auc"].append(roc)
        overall_results["precision"].append(prec)
        overall_results["recall"].append(rec)
        overall_results["svm_coefs"].append(best_params)

        all_folds_results.append([dataset_id, fold_id, acc, roc, prec, rec])

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, best_svm.decision_function(X_test_scaled))
        overall_results["roc_curve_data"].append((fpr, tpr))

    return overall_results, all_folds_results

###########################################
# Run K-Fold Cross-Validation on each dataset
###########################################

all_results = []
roc_data = []

all_y_tests = []
all_y_pred_probas = []

for dataset_id, feature_set in enumerate(data_list, start=1):
    print(f"\nProcessing Dataset {dataset_id}")
    X = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
    y = feature_set['Exacerbation.Outcome']
    top_features = all_shap_features_dict[dataset_id]

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold_id, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        X_train, X_test = X.iloc[train_index][top_features], X.iloc[test_index][top_features]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        best_svm = SVC(C=0.65, kernel='linear', probability=True, random_state=42, class_weight='balanced')
        best_svm.fit(X_train_scaled, y_train)
        y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]

        # Collect out-of-fold test labels and predicted probabilities
        all_y_tests.append(y_test.values)
        all_y_pred_probas.append(y_pred_proba)

    # Store fold results
    results, folds_results = k_fold_cross_validation(X, y, dataset_id, n_splits=10)

    # Store ROC curve data
    roc_data.extend(results["roc_curve_data"])

    # Store fold results
    all_results.extend(folds_results)

# **Create empty lists to store results**
all_results = []
all_shap_dfs = []
roc_data = []

for dataset_id, data in enumerate(data_list, 1):
    print(f"\nProcessing Dataset {dataset_id}")

    X = data.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
    y = data['Exacerbation.Outcome']

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold_id, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        print(f"Dataset {dataset_id}, Fold {fold_id}")

        # Always select only the top features for both train and test
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        top_features = all_shap_features_dict[dataset_id]
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_top)
        X_test_scaled = scaler.transform(X_test_top)

        # Grid search and fit
        param_grid = {
            'C': [0.0001, 0.0005, 0.001],
            'kernel': ['linear']
        }
        grid = GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_params = grid.best_params_
        print(f"Dataset {dataset_id} Fold {fold_id} best params: {best_params}")

        best_svm = SVC(**best_params, probability=True, random_state=42)
        best_svm.fit(X_train_scaled, y_train)

        # Metrics
        y_pred = best_svm.predict(X_test_scaled)
        y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        all_results.append([dataset_id, fold_id, acc, roc, prec, rec])

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, best_svm.decision_function(X_test_scaled))
        roc_data.append((fpr, tpr))

    # Compute mean ROC curve
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

    print(f"\nFinal Mean ROC-AUC for Dataset {dataset_id}: {mean_auc:.4f} ± {std_auc:.4f}")

    # Plot the average confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(np.round(cm).astype(int), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

# Convert results to DataFrame
df_results = pd.DataFrame(all_results, columns=["Dataset", "Fold", "Test Accuracy", "ROC-AUC", "Precision", "Recall"])
print(df_results['Test Accuracy'].mean())
print(df_results['ROC-AUC'].mean())

# Save Results
df_results.to_csv(r"corrected_5datasets_10fold_SVM_results.csv", index=False)



# Calculate overall metrics with std
print("\nOverall Metrics:")
print(f"Mean Test Accuracy: {df_results['Test Accuracy'].mean():.4f} ± {df_results['Test Accuracy'].std():.4f}")
print(f"Mean ROC-AUC Score: {df_results['ROC-AUC'].mean():.4f} ± {df_results['ROC-AUC'].std():.4f}")
print(f"Mean Precision: {df_results['Precision'].mean():.4f} ± {df_results['Precision'].std():.4f}")
print(f"Mean Recall: {df_results['Recall'].mean():.4f} ± {df_results['Recall'].std():.4f}")

# Calculate metrics per dataset
print("\nPer Dataset Performance:")
for dataset_id in range(1, 6):
    dataset_results = df_results[df_results['Dataset'] == dataset_id]
    
    print(f"\nDataset {dataset_id} Performance:")
    print(f"Test Accuracy: {dataset_results['Test Accuracy'].mean():.4f} ± {dataset_results['Test Accuracy'].std():.4f}")
    print(f"ROC-AUC Score: {dataset_results['ROC-AUC'].mean():.4f} ± {dataset_results['ROC-AUC'].std():.4f}")
    print(f"Precision: {dataset_results['Precision'].mean():.4f} ± {dataset_results['Precision'].std():.4f}")
    print(f"Recall: {dataset_results['Recall'].mean():.4f} ± {dataset_results['Recall'].std():.4f}")

all_metrics = []
all_conf_matrices = []
all_train_accs, all_val_accs = [], []
all_train_hinges, all_val_hinges = [], []

for dataset_id, feature_set in enumerate(data_list, 1):
    X = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
    y = feature_set['Exacerbation.Outcome']
    top_features = all_shap_features_dict[dataset_id]
    X_top = X[top_features]
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)

    # Grid search
    param_grid = {
        'C': [0.0001, 0.0005, 0.001],
        'kernel': ['linear']
    }
    grid = GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print(f"Dataset {dataset_id} best params: {best_params}")

    # Train with best params
    best_svm = SVC(**best_params, probability=True, random_state=42)
    best_svm.fit(X_train, y_train)

    # Metrics
    y_pred = best_svm.predict(X_test)
    y_pred_proba = best_svm.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    all_metrics.append([acc, prec, rec, f1, roc])
    all_conf_matrices.append(cm)

    # Overfitting/underfitting diagnostics
    y_train_pred = best_svm.predict(X_train)
    y_train_proba = best_svm.decision_function(X_train)
    y_test_proba = best_svm.decision_function(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = acc
    train_hinge = hinge_loss(y_train, y_train_proba)
    val_hinge = hinge_loss(y_test, y_test_proba)
    all_train_accs.append(train_acc)
    all_val_accs.append(val_acc)
    all_train_hinges.append(train_hinge)
    all_val_hinges.append(val_hinge)

# Aggregate metrics
all_metrics = np.array(all_metrics)
print("\nAverage metrics across datasets:")
print(f"Accuracy: {all_metrics[:,0].mean():.4f} ± {all_metrics[:,0].std():.4f}")
print(f"Precision: {all_metrics[:,1].mean():.4f} ± {all_metrics[:,1].std():.4f}")
print(f"Recall: {all_metrics[:,2].mean():.4f} ± {all_metrics[:,2].std():.4f}")
print(f"F1: {all_metrics[:,3].mean():.4f} ± {all_metrics[:,3].std():.4f}")
print(f"ROC-AUC: {all_metrics[:,4].mean():.4f} ± {all_metrics[:,4].std():.4f}")

# Average confusion matrix
avg_cm = np.mean(np.array(all_conf_matrices), axis=0)
print("\nAverage Confusion Matrix (rounded):")
print(np.round(avg_cm).astype(int))

# Plot the average confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(np.round(avg_cm).astype(int), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Average Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()

# Overfitting/underfitting diagnostics
print("\nTrain vs Validation Accuracy:")
print(f"Train: {np.mean(all_train_accs):.4f} ± {np.std(all_train_accs):.4f}")
print(f"Val:   {np.mean(all_val_accs):.4f} ± {np.std(all_val_accs):.4f}")
print("\nTrain vs Validation Hinge Loss:")
print(f"Train: {np.mean(all_train_hinges):.4f} ± {np.std(all_train_hinges):.4f}")
print(f"Val:   {np.mean(all_val_hinges):.4f} ± {np.std(all_val_hinges):.4f}")

# Save out-of-fold test labels and predicted probabilities
y_true_svm = np.concatenate(all_y_tests)
y_pred_proba_svm = np.concatenate(all_y_pred_probas)
np.save('svm_y_true.npy', y_true_svm)
np.save('svm_y_pred_proba.npy', y_pred_proba_svm)

# Report K-Fold metrics and plot ROC
print("\nK-Fold CV Metrics (out-of-fold):")
print(f"Accuracy: {accuracy_score(y_true_svm, (y_pred_proba_svm > 0.5).astype(int)):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_true_svm, y_pred_proba_svm):.4f}")
print(f"Precision: {precision_score(y_true_svm, (y_pred_proba_svm > 0.5).astype(int)):.4f}")
print(f"Recall: {recall_score(y_true_svm, (y_pred_proba_svm > 0.5).astype(int)):.4f}")
print(f"F1: {f1_score(y_true_svm, (y_pred_proba_svm > 0.5).astype(int)):.4f}")

fpr, tpr, _ = roc_curve(y_true_svm, y_pred_proba_svm)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM (K-Fold CV)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve - SVM (K-Fold CV)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print(f'Mean ROC-AUC from folds: {mean_auc:.4f} ± {std_auc:.4f}')



#======================================================
# June 30, ROC
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
plt.title('Mean ROC Curve - SVM (K-Fold CV)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


np.save('svm_mean_fpr_Gene_200.npy', mean_fpr)
np.save('svm_mean_tpr_Gene_200.npy', mean_tpr)


mean_fpr = np.load('svm_mean_fpr_Gene_200.npy')
mean_tpr = np.load('svm_mean_tpr_Gene_200.npy')

plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve - SVM')
plt.legend()
plt.show()
