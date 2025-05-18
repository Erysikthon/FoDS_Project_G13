from sklearn.model_selection import train_test_split
from Data_Overview import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import os
from sklearn.model_selection import KFold




#Transforming object col into category
#data[cat_features] = data[cat_features].astype('category')

####################### SPLITTING DATA ####################################################
X = data[features]
y = data[label]

yeares = data["JAHR"]

#for all years
if current_year == "all":
    stratify_col = y.astype(str) + "_" + yeares.astype(str)

else:
    stratify_col = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify= stratify_col)  #for all years also stratify years


############################  MISSING DATA HANDLING ###########################################

#Filling Categorical Columns
for n in cat_features:
    X_train[n] = X_train[n].fillna("NA")
    X_test[n] = X_test[n].fillna("NA")

#Filling Numeric Columns
for i in X_train[num_features]:
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_test[i] = X_test[i].fillna(X_train[i].mean())      # To ensure consistency with X_train's mean


X_train[cat_features] = X_train[cat_features].astype('category')
X_test[cat_features] = X_test[cat_features].astype('category')

######################### ONE HOT ENCODING ##################################################################
"""
X_train = pd.get_dummies(X_train, columns=cat_features, drop_first=True)

X_test = pd.get_dummies(X_test, columns=cat_features, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
"""

########################## ALTERNATIVE -> TARGET ENCODING ##########################################################
def target_encode_feature(train_series, test_series, target, n_splits=5, alpha=5):
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Converting series to numpy arrays to avoid categorical dtype issues
    train_encoded = np.zeros(len(train_series))
    test_encoded = np.zeros(len(test_series))

    # Global mean for smoothing and handling unseen categories
    global_mean = target.mean()

    # k-fold target encoding
    for train_idx, val_idx in kf.split(train_series):
        # Get current fold data
        fold_train = train_series.iloc[train_idx]
        fold_val = train_series.iloc[val_idx]
        fold_target = target.iloc[train_idx]

        # Calculating means for each category with smoothing
        category_means = {}
        for category in fold_train.unique():
            cat_idx = fold_train == category
            n = cat_idx.sum()
            cat_mean = fold_target[cat_idx].mean()
            # Apply smoothing
            category_means[category] = (cat_mean * n + global_mean * alpha) / (n + alpha)

        #Apply encoding
        for category, mean_value in category_means.items():
            train_encoded[val_idx[fold_val == category]] = mean_value
        # Fill unseen categories with global mean
        train_encoded[val_idx[~fold_val.isin(category_means.keys())]] = global_mean

    # Encode test set using all training data
    category_means = {}
    for category in train_series.unique():
        cat_idx = train_series == category
        n = cat_idx.sum()
        cat_mean = target[cat_idx].mean()
        category_means[category] = (cat_mean * n + global_mean * alpha) / (n + alpha)

    # Apply encoding to test set
    for category, mean_value in category_means.items():
        test_encoded[test_series == category] = mean_value
    # Fill unseen categories with global mean
    test_encoded[~test_series.isin(category_means.keys())] = global_mean

    return train_encoded, test_encoded


# Apply target encoding to each categorical feature
for feature in cat_features:
    X_train[feature], X_test[feature] = target_encode_feature(
        X_train[feature],
        X_test[feature],
        y_train
    )

#######################  SCALING (after splitting!!) ###############################################################
sc = StandardScaler()
X_train[num_features] = sc.fit_transform(X_train[num_features])
X_test[num_features]  = sc.transform(X_test[num_features])


##################### EVALUATION METRICS ###############################################################################
def eval_Performance(y_eval, X_eval, clf, clf_name = 'My Classifier'):

    y_pred = clf.predict(X_eval)
    y_pred_proba = clf.predict_proba(X_eval)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    # Evaluation
    accuracy  = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall    = recall_score(y_eval, y_pred)
    f1        = f1_score(y_eval, y_pred)
    fp_rates, tp_rates, _ = roc_curve(y_eval, y_pred_proba)

    #Area under the roc curve
    roc_auc = auc(fp_rates, tp_rates)

    return tp,fp,tn,fn,accuracy, precision, recall, f1, roc_auc

df_performance = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )



##################### MODEL TRAINING ##################################################################################
clf_LR = LogisticRegression(random_state=10, class_weight='balanced')
clf_LR.fit(X_train, y_train)

df_performance.loc['LR (test)',:] = eval_Performance(y_test, X_test, clf_LR, clf_name ='LR')
df_performance.loc['LR (train)',:] = eval_Performance(y_train, X_train, clf_LR, clf_name ='LR (train)')

print("Total number of features (LR):", X_train.shape[1])


##################### FEATURE SELECTION ###############################################################################

#Univariate FS
UVFS_Selector = SelectKBest(score_func = f_classif, k=20)

X_UVFS = UVFS_Selector.fit_transform(X_train, y_train)
X_UVFS_test = UVFS_Selector.transform(X_test)

# Scores
scores = UVFS_Selector.scores_  #ANOVA scores
pvalues = UVFS_Selector.pvalues_  #p-values for each feature


UVFS_selected_features = UVFS_Selector.get_feature_names_out(input_features=X_train.columns)
print("Selected Features (UVFS):", UVFS_selected_features)


# Visualization of top selected features' importance
X_indices = np.arange(len(UVFS_selected_features))  # Use the range of selected features
plt.figure(figsize=(10, 5))
plt.bar(X_indices, scores[:len(UVFS_selected_features)], width=0.5, align='center', alpha=0.8, color='orange')
plt.title("Feature Univariate Score (ANOVA F-statistic)")
plt.xlabel("Features")
plt.ylabel("p_{value}")
plt.xticks(X_indices, UVFS_selected_features, rotation=90)  # Use only selected features' names
plt.tight_layout()
plt.show()



# Evaluate performance using only k top features
clf_LR_UVFS = LogisticRegression(random_state=10, class_weight='balanced')
clf_LR_UVFS.fit(X_UVFS, y_train)

#Coefficients of top k features
coefficients = clf_LR_UVFS.coef_[0]
for feature, coef in zip(UVFS_selected_features, coefficients):
    print(f"Feature: {feature}, Coefficient: {coef}")

# Evaluation
df_performance.loc['LR (test,UVFS)',:] = eval_Performance(y_test, X_UVFS_test, clf_LR_UVFS, clf_name = 'LR_UVFS')
df_performance.loc['LR (train,UVFS)',:] = eval_Performance(y_train, X_UVFS, clf_LR_UVFS, clf_name = 'LR_UVFS (train)')
print(df_performance)


#L1 REGULARIZATION
param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],  # Regularization strength
    "penalty": ["l1"],  # L1 regularization
    "solver": ["liblinear"]  # Solver for logistic regression
}
clf_LR_L1 = GridSearchCV(
    LogisticRegression(random_state=10, class_weight='balanced'),
    param_grid,
    cv=5 ,
    #scoring='balanced_accuracy',
    scoring ='roc_auc'
)

# Fit GridSearchCV to training data
clf_LR_L1.fit(X_train, y_train)

# Best L1 model after hyperparameter tuning
best_L1_model = clf_LR_L1.best_estimator_

n_used_features = np.sum(best_L1_model.coef_ != 0)
print(f"Number of features used (L1): {n_used_features}")

print("Best parameters (L1):", clf_LR_L1.best_params_)

# Feature Importance
coefs = best_L1_model.coef_[0]  # Get coefficients from the model
coef_L1 = pd.DataFrame({
    'Feature': X_train.columns,  # Using feature names from X_train
    'Importance': np.abs(coefs)
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features (L1):")
print(coef_L1.head(10))  #

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=coef_L1.head(10))
plt.title("Top 10 Feature Importances (L1)")
plt.tight_layout()

file_name = 'L1_features_top_10.png'
file_path = os.path.join("output", file_name)
plt.savefig(file_path, dpi=300)


#Evaluation of performance on train and test sets
df_performance.loc['LR (test,L1)', :] = eval_Performance(y_test, X_test, best_L1_model, clf_name='LR_L1')
df_performance.loc['LR (train,L1)', :] = eval_Performance(y_train, X_train, best_L1_model,
                                                              clf_name='LR_L1 (train)')


#Extract selected features
L1_selected_features = []
for col, coef in zip(X.columns, best_L1_model.coef_[0]):
    if coef != 0:  # Keep only features with non-zero coefficients
        L1_selected_features.append(col)
print(f"Selected features: {L1_selected_features}")




############################## L2 REGULARIZATION ###########################################################

param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],  # Regularization strength
    "penalty": ["l2"],  #L2 regularization
    "solver": ["liblinear"]  # Solver for logistic regression
}
clf_LR_L2 = GridSearchCV(
    LogisticRegression(random_state=10, class_weight='balanced'),
    param_grid,
    cv=5,
    #scoring='balanced_accuracy'
    scoring='roc_auc'
)

clf_LR_L2.fit(X_train, y_train)

best_L2_model = clf_LR_L2.best_estimator_
n_used_features_L2 = np.sum(best_L2_model.coef_ != 0)
print(f"Number of features used (nonzero weights): {n_used_features_L2}")

print("Best parameters (L2):", clf_LR_L2.best_params_)

# Feature Importance
coefs = best_L2_model.coef_[0]  # Get coefficients from the model
coef_L1 = pd.DataFrame({
    'Feature': X_train.columns,  # Using feature names from X_train
    'Importance': np.abs(coefs)
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features (L1):")
print(coef_L1.head(10))  #

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=coef_L1.head(10))
plt.title("Top 10 Feature Importances (L2)")
plt.tight_layout()

file_name = 'L2_features_top_10.png'
file_path = os.path.join("output", file_name)
plt.savefig(file_path, dpi=300)



# Evaluation of the reduced feature model
df_performance.loc['LR (test,L2)', :] = eval_Performance(y_test, X_test, clf_LR_L2,
                                                              clf_name='LR_L2')
df_performance.loc['LR (train,L2)', :] = eval_Performance(y_train, X_train, clf_LR_L2,
                                                               clf_name='LR_L2(train)')


#Coefficients of selected features

#for feature, coef in zip(L1_selected_features, best_L2_model.coef_[0]):
#    print(f"Feature: {feature}, Coefficient: {coef}")


########################## Recursive Feature Eliminiation with Cross-Validation #####################################

log_reg = LogisticRegression(class_weight='balanced', solver="liblinear", random_state=10)

rfe_cv = RFECV(estimator=log_reg, step=1, cv=5, scoring='roc_auc')
rfe_cv.fit(X_train, y_train)



df_performance.loc['LR (test,RFE_CV)', :] = eval_Performance(y_test, X_test, rfe_cv,
                                                             clf_name='LR_rfe_cv')
df_performance.loc['LR (train,RFE_CV)', :] = eval_Performance(y_train, X_train, rfe_cv,
                                                              clf_name='LR_rfe_cv (train)')

# Print optimal number of features
print(f"Optimal number of features (RFE): {rfe_cv.n_features_}")

# Get the features selected
rfe_selected_features = X_train.columns[rfe_cv.support_]
print(f"Selected features: {list(rfe_selected_features)}")

# Feature Importance
coefs = rfe_cv.estimator_.coef_[0]  # coefficients from the fitted estimator
coef_rfe = pd.DataFrame({
    'Feature': rfe_selected_features,
    'Importance': np.abs(coefs)  # Only use coefficients for selected features
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features (RFE):")
print(coef_rfe.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=coef_rfe.head(10))
plt.title("Top 10 Feature Importances (RFE)")
plt.tight_layout()

file_name = 'RFE_features_top_10.png'
file_path = os.path.join("output", file_name)
plt.savefig(file_path, dpi=300)



print(df_performance)



##################################### PLOTS ############################################################################
os.makedirs('output', exist_ok=True)

#Accuracy
df_performance[['accuracy']].plot(kind='bar', figsize=(15, 12))
plt.title('Performance Comparison of Models with All vs Selected Features')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim([0, 1])
plt.grid(True, axis='y', linestyle='--')
plt.show()
#plt.savefig('../output/LR_Accuracy.png')


#F1-Score
df_performance[['f1']].plot(kind='bar', figsize=(15, 12), color='teal')
plt.title('F1-Score Comparison of Models with All vs Selected Features')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)
plt.ylim([0, 1])  # F1-Score ranges between 0 and 1
plt.grid(True, axis='y', linestyle='--')
plt.show()

#ROC-AUC BAR PLOT
df_performance[['roc_auc']].plot(kind='bar', figsize=(15, 12), color='orange')
plt.title('ROC-AUC Comparison of Models with All vs Selected Features')
plt.ylabel('ROC-AUC')
plt.xticks(rotation=45)
plt.ylim([0.5, 1])  # ROC-AUC ranges between 0.5 (random) and 1 (perfect)
plt.grid(True, axis='y', linestyle='--')
plt.show()

#ROC-AUC Plot for Test SEt
plt.figure(figsize=(10, 7))

# Logistic Regression (All Features)
y_score_lr = clf_LR.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, lw=2, label=f'LogReg (AUC = {roc_auc_lr:.2f})')

# Logistic Regression with Univariate Feature Selection
y_score_uvfs = clf_LR_UVFS.predict_proba(X_UVFS_test)[:, 1]
fpr_uvfs, tpr_uvfs, _ = roc_curve(y_test, y_score_uvfs)
roc_auc_uvfs = auc(fpr_uvfs, tpr_uvfs)
plt.plot(fpr_uvfs, tpr_uvfs, lw=2, label=f'LogReg+UVFS (AUC = {roc_auc_uvfs:.2f})')

# Logistic Regression with L1 Regularization
y_score_l1 = best_L1_model.predict_proba(X_test)[:, 1]
fpr_l1, tpr_l1, _ = roc_curve(y_test, y_score_l1)
roc_auc_l1 = auc(fpr_l1, tpr_l1)
plt.plot(fpr_l1, tpr_l1, lw=2, label=f'LogReg+L1 (AUC = {roc_auc_l1:.2f})')

# Logistic Regression with L2 Regularization
y_score_l2 = best_L2_model.predict_proba(X_test)[:, 1]
fpr_l2, tpr_l2, _ = roc_curve(y_test, y_score_l2)
roc_auc_l2 = auc(fpr_l2, tpr_l2)
plt.plot(fpr_l2, tpr_l2, lw=2, label=f'LogReg+L2 (AUC = {roc_auc_l2:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("output/LR_ROC_Curve.png")


#Precision
df_performance[['precision']].plot(kind='bar', figsize=(15, 12), color='green')
plt.title('Precision Comparison of Models with All vs Selected Features')
plt.ylabel('Precision')
plt.xticks(rotation=45)
plt.ylim([0, 1])
plt.grid(True, axis='y', linestyle='--')
plt.show()



