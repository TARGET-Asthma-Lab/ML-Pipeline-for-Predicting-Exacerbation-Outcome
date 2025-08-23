
from sklearn.metrics import confusion_matrix, log_loss
from scipy.stats import chi2
import statsmodels.api as sm
import numpy as np 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, roc_curve, auc
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score


# Precompute SHAP Features for All Datasets (Before Cross-Validation)
all_shap_features_dict = {}
shap_feature_dfs = []

for dataset_id, feature_set in enumerate(merged_data_list, 1):
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

# # Save Precomputed SHAP Features
# shap_features_df = pd.concat(shap_feature_dfs, ignore_index=True)
# shap_features_df.to_csv(r"C:\DATA\bayesian\data\interim\Preselected_SHAP_Features_LogReg_CenClinDiv_200_May27.csv", index=False)
# print("SHAP Features Precomputed and Saved!")


# Train and evaluate Logistic Regression
def train_logistic_regression(X, y, top_features):
    """Trains Logistic Regression on selected SHAP features and evaluates performance."""
    X_top = X[top_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(penalty='l2', solver='liblinear', C=0.01, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    train_logloss = log_loss(y_train, model.predict_proba(X_train)[:, 1])
    val_logloss = log_loss(y_test, model.predict_proba(X_test)[:, 1])
    coef_dict = dict(zip(top_features, model.coef_[0]))
    return model, accuracy, roc_auc, precision, recall, f1, coef_dict, y_test, y_pred_proba, y_pred, train_logloss, val_logloss

def hosmer_lemeshow_test(y_true, y_pred_proba, g=10):
    data = np.array(list(zip(y_true, y_pred_proba)))
    data = data[data[:,1].argsort()]
    n = len(y_true)
    group_size = n // g
    hl_stat = 0
    for i in range(g):
        start = i * group_size
        end = (i+1) * group_size if i < g-1 else n
        group = data[start:end]
        obs = np.sum(group[:,0])
        exp = np.sum(group[:,1])
        n_group = end - start
        if exp > 0 and exp < n_group:
            hl_stat += (obs - exp)**2 / (exp * (1 - exp/n_group))
    p_value = chi2.sf(hl_stat, g-2)
    return hl_stat, p_value

# K-Fold Cross-Validation Function
def k_fold_cross_validation(X, y, dataset_id, n_splits=10):
    """Performs 10-fold cross-validation using precomputed SHAP-selected features."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    overall_results = {
        "accuracy": [], "roc_auc": [], "precision": [], "recall": [],
        "roc_curve_data": [], "logreg_coefs": [],
        "deviance": [], "null_deviance": [], "chi2_stat": [], "p_value": [],
        "hl_stat": [], "hl_p": [], "cms": []
    }
    all_folds_results = []

    # ADD THESE LINES
    all_y_tests = []
    all_y_pred_probas = []

    for fold_id, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        print(f"Dataset {dataset_id}, Fold {fold_id}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Use Precomputed SHAP Features
        top_features = all_shap_features_dict[dataset_id]

        # Train Logistic Regression and compute metrics
        _, accuracy, roc_auc, precision, recall, f1, coef_dict, y_test, y_pred_proba, y_pred, train_logloss, val_logloss = train_logistic_regression(X_train, y_train, top_features)

        # ADD THESE LINES
        all_y_tests.append(y_test)
        all_y_pred_probas.append(y_pred_proba)

        # Store fold results
        overall_results["accuracy"].append(accuracy)
        overall_results["roc_auc"].append(roc_auc)
        overall_results["precision"].append(precision)
        overall_results["recall"].append(recall)
        overall_results["logreg_coefs"].append(coef_dict)

        all_folds_results.append([dataset_id, fold_id, accuracy, roc_auc, precision, recall])

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        overall_results["roc_curve_data"].append((fpr, tpr))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        # Deviance
        ll_model = -log_loss(y_test, y_pred_proba, normalize=False)
        deviance = -2 * ll_model

        # Null deviance (fit intercept only)
        mean_prob = np.mean(y_test)
        ll_null = -log_loss(y_test, np.full_like(y_test, mean_prob), normalize=False)
        null_deviance = -2 * ll_null

        # Chi-square goodness-of-fit
        chi2_stat = null_deviance - deviance
        df = len(y_test) - 2  # n - number of parameters (intercept + 1 for binary)
        p_value = chi2.sf(chi2_stat, df)

        # Hosmer-Lemeshow test (using statsmodels)
        hl_test = sm.stats.diagnostic.acorr_ljungbox(y_test - y_pred_proba, lags=[10], return_df=True)
        # Or use statsmodels.stats.diagnostic.linear_harvey_collier if you want a different test

        # Store or print these as needed
        print(f"Confusion Matrix:\n{cm}")
        print(f"Deviance: {deviance:.4f}, Null Deviance: {null_deviance:.4f}, Chi2: {chi2_stat:.4f}, p-value: {p_value:.4g}")
        print(f"Hosmer-Lemeshow (Ljung-Box) p-value: {hl_test['lb_pvalue'].iloc[0]:.4g}")

        hl_stat, hl_p = hosmer_lemeshow_test(y_test, y_pred_proba, g=10)
        print(f"Hosmer-Lemeshow stat: {hl_stat:.4f}, p-value: {hl_p:.4g}")

        # Append metrics to overall_results
        overall_results["deviance"].append(deviance)
        overall_results["null_deviance"].append(null_deviance)
        overall_results["chi2_stat"].append(chi2_stat)
        overall_results["p_value"].append(p_value)
        overall_results["hl_stat"].append(hl_stat)
        overall_results["hl_p"].append(hl_p)
        overall_results["cms"].append(cm)

    return overall_results, all_folds_results, all_y_tests, all_y_pred_probas

# Run K-Fold Cross-Validation on each dataset
all_results = []
roc_data = []

# Collect all overall_results from each dataset
all_overall_results = []

for dataset_id, feature_set in enumerate(merged_data_list, start=1):
    print(f"\nProcessing Dataset {dataset_id}")
    X, y = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore'), feature_set['Exacerbation.Outcome']
    results, folds_results, all_y_tests, all_y_pred_probas = k_fold_cross_validation(X, y, dataset_id, n_splits=10)
    all_overall_results.append(results)

    # Store ROC curve data
    roc_data.extend(results["roc_curve_data"])

    # Store fold results
    all_results.extend(folds_results)

# Compute mean ROC curve
mean_fpr_ClinGenDivLR_200 = np.linspace(0, 1, 100)
mean_tpr_ClinGenDivLR_200 = np.mean([np.interp(mean_fpr_ClinGenDivLR_200, fpr, tpr) for fpr, tpr in roc_data], axis=0)
std_tpr_ClinGenDivLR_200 = np.std([np.interp(mean_fpr_ClinGenDivLR_200, fpr, tpr) for fpr, tpr in roc_data], axis=0)
final_auc_ClinGenDivLR_200 = auc(mean_fpr_ClinGenDivLR_200, mean_tpr_ClinGenDivLR_200)

print(f"\nFinal Mean ROC-AUC across datasets: {final_auc_ClinGenDivLR_200:.4f}")

# Convert results to DataFrame
df_results = pd.DataFrame(all_results, columns=["Dataset", "Fold", "Test Accuracy", "ROC-AUC", "Precision", "Recall"])
print(f"\nOverall Mean Accuracy: {df_results['Test Accuracy'].mean():.4f}")
print(f"Overall Mean ROC-AUC: {df_results['ROC-AUC'].mean():.4f}")

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

# Example: after all folds for one dataset
def print_avg_metrics(overall_results):
    print(f"Mean Deviance: {np.mean(overall_results['deviance']):.4f} ± {np.std(overall_results['deviance']):.4f}")
    print(f"Mean Null Deviance: {np.mean(overall_results['null_deviance']):.4f} ± {np.std(overall_results['null_deviance']):.4f}")
    print(f"Mean Chi2 Stat: {np.mean(overall_results['chi2_stat']):.4f} ± {np.std(overall_results['chi2_stat']):.4f}")
    print(f"Mean p-value: {np.mean(overall_results['p_value']):.4g} ± {np.std(overall_results['p_value']):.4g}")
    print(f"Mean HL Stat: {np.mean(overall_results['hl_stat']):.4f} ± {np.std(overall_results['hl_stat']):.4f}")
    print(f"Mean HL p-value: {np.mean(overall_results['hl_p']):.4g} ± {np.std(overall_results['hl_p']):.4g}")

    # Average confusion matrix
    avg_cm = np.mean(np.array(overall_results['cms']), axis=0)
    print("Average Confusion Matrix (rounded):")
    print(np.round(avg_cm).astype(int))

# Aggregate metrics across all datasets
def aggregate_metric(metric_name):
    return np.concatenate([np.array(r[metric_name]) for r in all_overall_results])

def aggregate_cms():
    all_cms = np.concatenate([np.array(r["cms"])[None, ...] for r in all_overall_results], axis=0)
    return np.mean(all_cms, axis=(0, 1))

# Print average metrics
print("\n=== Average Metrics Across All Datasets and Folds ===")
for metric in ["deviance", "null_deviance", "chi2_stat", "p_value", "hl_stat", "hl_p"]:
    values = aggregate_metric(metric)
    print(f"Mean {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# Average confusion matrix
avg_cm = aggregate_cms()
print("\nAverage Confusion Matrix (rounded):")
print(np.round(avg_cm).astype(int))

# Plot average confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(np.round(avg_cm).astype(int), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Average Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()

# # --- SGDRegressor: Average Training/Validation MSE with std ---
# def sgd_regressor_epoch_metrics(X, y, n_epochs=20, val_size=0.2, random_state=42, loss='squared_error', penalty='l2', alpha=0.0001, eta0=0.01):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=val_size, random_state=random_state)
#     model = SGDRegressor(
#         loss=loss,
#         penalty=penalty,
#         alpha=alpha,
#         max_iter=1,
#         learning_rate='constant',
#         eta0=eta0,
#         warm_start=True,
#         random_state=random_state
#     )
#     train_mse, val_mse = [], []
#     for epoch in range(n_epochs):
#         model.fit(X_train, y_train)
#         y_train_pred = model.predict(X_train)
#         y_val_pred = model.predict(X_val)
#         train_mse.append(mean_squared_error(y_train, y_train_pred))
#         val_mse.append(mean_squared_error(y_val, y_val_pred))
#     return train_mse, val_mse

all_train_loglosses = []
all_val_loglosses = []
all_coef_dfs = []  

for dataset_id, feature_set in enumerate(merged_data_list, 1):
    X = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
    y = feature_set['Exacerbation.Outcome']
    top_features = all_shap_features_dict[dataset_id]
    X_top = X[top_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(penalty='l2', solver='liblinear', C=0.01, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    train_logloss = log_loss(y_train, model.predict_proba(X_train)[:, 1])
    val_logloss = log_loss(y_test, model.predict_proba(X_test)[:, 1])
    all_train_loglosses.append(train_logloss)
    all_val_loglosses.append(val_logloss)
    
    # Get coefficients and corresponding features
    coefs = model.coef_[0]
    feature_names = top_features
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coef': coefs
    })
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    coef_df['dataset_id'] = dataset_id
    all_coef_dfs.append(coef_df)  # <-- collect each DataFrame

# After the loop, concatenate and save
all_coef_df = pd.concat(all_coef_dfs, ignore_index=True)
# all_coef_df.to_csv(r"C:\DATA\bayesian\data\interim\Coef_LogReg_Trained_CenClinDiv_200_June3.csv", index=False)

print(f"Average Train Log-Loss: {np.mean(all_train_loglosses):.4f} ± {np.std(all_train_loglosses):.4f}")
print(f"Average Validation Log-Loss: {np.mean(all_val_loglosses):.4f} ± {np.std(all_val_loglosses):.4f}")

# --- Collect metrics across all datasets ---
all_accuracies = []
all_roc_aucs = []
all_precisions = []
all_recalls = []
all_f1s = []

for dataset_id, feature_set in enumerate(merged_data_list, 1):
    X = feature_set.drop(columns=['Exacerbation.Outcome', 'subject_id'], errors='ignore')
    y = feature_set['Exacerbation.Outcome']
    top_features = all_shap_features_dict[dataset_id]
    _, accuracy, roc_auc, precision, recall, f1, _, _, _, _, train_logloss, val_logloss = train_logistic_regression(X, y, top_features)
    all_accuracies.append(accuracy)
    all_roc_aucs.append(roc_auc)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)

print(f"Average Test Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
print(f"Average ROC-AUC: {np.mean(all_roc_aucs):.4f} ± {np.std(all_roc_aucs):.4f}")
print(f"Average Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
print(f"Average Recall: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
print(f"Average F1: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}") 





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
plt.title('Mean ROC Curve - LR (K-Fold CV)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


np.save('lr_mean_fpr_Gene_200.npy', mean_fpr)
np.save('lr_mean_tpr_Gene_200.npy', mean_tpr)


mean_fpr = np.load('lr_mean_fpr_Gene_200.npy')
mean_tpr = np.load('lr_mean_tpr_Gene_200.npy')

plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve - LR')
plt.legend()
plt.show()



# all test labels and predicted probabilities from all folds:
# all_y_tests, all_y_pred_probas

# If data as lists, flatten them:
y_true_logreg = np.concatenate([np.array(y) for y in all_y_tests])
y_pred_proba_logreg = np.concatenate([np.array(y) for y in all_y_pred_probas])

# Save to disk
np.save('logreg_y_true_Gen_200.npy', y_true_logreg)
np.save('logreg_y_pred_proba_Gen_200.npy', y_pred_proba_logreg)

# Optionally, plot ROC here as well
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_true_logreg, y_pred_proba_logreg)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'LogReg (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()  
