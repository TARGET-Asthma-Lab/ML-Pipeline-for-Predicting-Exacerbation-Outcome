### 1) Logistic Regression

### Gene only
``python src/models/logisticregression.py \
  --data  "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_logreg_k60``

### Clinical only
python src/models/logisticregression.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clinical_logreg_k60

### ClinGen (premerged gene+clinical)
python src/models/logisticregression.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_logreg_k60

### Exacerbation-only
python src/models/logisticregression.py \
  --data  "data/exacer/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_ExacerOut.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag exacer_logreg_k60

### Clinical + α + β diversity
python src/models/logisticregression.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clin_ab_logreg_k60

### ClinGen + α + β
python src/models/logisticregression.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_logreg_k60

### Gene + Raw diversity (PCA 95%)
python src/models/logisticregression.py \
  --data     "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --raw-div  "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_raw_logreg_k60

### ClinGen + α + β + Raw diversity (PCA 95%)
python src/models/logisticregression.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --raw-div "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_raw_logreg_k60

### 2) SVM (RBF kernel)

### Gene only
python src/models/svm.py \
  --data  "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_svm_k60

### Clinical only
python src/models/svm.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clinical_svm_k60

### ClinGen (premerged gene+clinical)
python src/models/svm.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_svm_k60

### Exacerbation-only
python src/models/svm.py \
  --data  "data/exacer/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_ExacerOut.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag exacer_svm_k60

### Clinical + α + β diversity
python src/models/svm.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clin_ab_svm_k60

### ClinGen + α + β
python src/models/svm.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_svm_k60

### Gene + Raw diversity (PCA 95%)
python src/models/svm.py \
  --data     "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --raw-div  "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_raw_svm_k60

### ClinGen + α + β + Raw diversity (PCA 95%)
python src/models/svm.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --raw-div "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_raw_svm_k60

### 3) XGBoost

### Gene only
python src/models/xgboost.py \
  --data  "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_xgb_k60

### Clinical only
python src/models/xgboost.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clinical_xgb_k60

### ClinGen (premerged gene+clinical)
python src/models/xgboost.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_xgb_k60

### Exacerbation-only
python src/models/xgboost.py \
  --data  "data/exacer/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_ExacerOut.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag exacer_xgb_k60

### Clinical + α + β diversity
python src/models/xgboost.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clin_ab_xgb_k60

### ClinGen + α + β
python src/models/xgboost.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_xgb_k60

### Gene + Raw diversity (PCA 95%)
python src/models/xgboost.py \
  --data     "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --raw-div  "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_raw_xgb_k60

### ClinGen + α + β + Raw diversity (PCA 95%)
python src/models/xgboost.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --raw-div "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_raw_xgb_k60

### 4) Feedforward Neural Network (PyTorch)

### Gene only
python src/models/feedforward.py \
  --data  "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_ffnn_k60

### Clinical only
python src/models/feedforward.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clinical_ffnn_k60

### ClinGen (premerged gene+clinical)
python src/models/feedforward.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ffnn_k60

### Exacerbation-only
python src/models/feedforward.py \
  --data  "data/exacer/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_ExacerOut.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag exacer_ffnn_k60

### Clinical + α + β diversity
python src/models/feedforward.py \
  --data  "data/clinical/clinical_Oct30_imputed_rf_{i}_vv_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clin_ab_ffnn_k60

### ClinGen + α + β
python src/models/feedforward.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_ffnn_k60

### Gene + Raw diversity (PCA 95%)
python src/models/feedforward.py \
  --data     "data/gene/gene_RF_imputed_Oct30_{i}_sklearnImp_NoClinical.csv" \
  --raw-div  "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag gene_raw_ffnn_k60

### ClinGen + α + β + Raw diversity (PCA 95%)
python src/models/feedforward.py \
  --data  "data/clingen/gene_RF_imputed_Oct30_{i}_sklearnImp_Asthma_treatment_modified_Apr10.csv" \
  --alpha "data/alpha_div/alpha_div_imputed_pmm{i}_Jan30.csv" \
  --beta  "data/beta_div/beta_div_imputed_pmm{i}_Jan30.csv" \
  --raw-div "data/rawdiv/otutab_transp_div_imputed_fastFeb13.csv" \
  --raw-div-pca-var 0.95 \
  --runs 5 --target Exacerbation.Outcome --top-k 60 --tag clingen_ab_raw_ffnn_k60
To switch to 200 features, change --top-k 60 → --top-k 200 in any command.
