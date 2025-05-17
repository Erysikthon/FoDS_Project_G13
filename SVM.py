import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from Data_Preparation import df
from Data_Preparation import *
import os

# --- Parameters --------
tt_size = 0.2
n_ftrs = 10
#yr = 2011
model_type = "svc"  # Options: 'svc', 'hgb'


#data = df.copy()

md_choice = input('Which model do you prefer? (default is svc) ')

for i in md_choice.split(' '): 
    if i.startswith('20'):
        yer = int(i)
        print(yer)
        if yer > 2008 and yer < 2020:
            yr = yer
            data = df[df["JAHR"]==yr].copy()
            print('chosen_year', yr)
    elif i == 'svc':
        model_type = i
    elif i == 'hgb':
        model_type = i
    elif i.startswith('0.'): #testsplit
        tt_size = float(i)
        print('chosen test-split size:', tt_size)


# --- Load and prepare data --------
#data = df.copy()
#data = df[df["JAHR"]==yr].copy()

"""
label = "FiErg"
features = ["KT", "Inst", "Adr", "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand",
            "SA", "PtageStatT", "AustStatT", "NeugStatT", "Ops", "Gebs", "CMIb", "CMIn",
            "pPatWAU", "pPatWAK", "pPatLKP", "pPatHOK", "PersA", "PersP", "PersMT", "PersT",
            "PersAFall", "PersPFall", "PersMTFall", "PersTFall", "AnzBelA", "AnzBelP (nur ab KZP2010)"]

data = data.dropna(subset=[label])  # Drop missing labels

# Drop all-NaN columns
columns_to_drop = [col for col in features if data[col].isna().sum() == len(data[col])]
data.drop(columns=columns_to_drop, inplace=True)
features = [col for col in features if col not in columns_to_drop]

# Rename columns
if "AnzBelP (nur ab KZP2010)" in data.columns:
    data.rename(columns={"AnzBelP (nur ab KZP2010)": "AnzBelP_KZP2010"}, inplace=True)
    features = ["AnzBelP_KZP2010" if col == "AnzBelP (nur ab KZP2010)" else col for col in features]

# Binary target
data[label] = (data[label] > 0).astype(int)
"""

# Split
X = data[features]
y = data[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tt_size, random_state=42) #stratifying ???????????????????????????????????????????????????????????

# Feature types
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# --- Preprocessing and modeling ---

if model_type == "svc":
    # --- Numerical preprocessing ---
    num_imputer = SimpleImputer(strategy="mean")
    X_train_num = num_imputer.fit_transform(X_train[numerical_features])
    X_test_num = num_imputer.transform(X_test[numerical_features])

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # --- Categorical preprocessing ---
    cat_imputer = SimpleImputer(strategy="constant", fill_value=None)
    X_train_cat = cat_imputer.fit_transform(X_train[categorical_features])
    X_test_cat = cat_imputer.transform(X_test[categorical_features])

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat_encoded = onehot.fit_transform(X_train_cat)
    X_test_cat_encoded = onehot.transform(X_test_cat)

    # --- Combine numeric + categorical ---
    X_train_processed = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_num_scaled, X_test_cat_encoded])

    # Drop NaNs that couldn't be handled
    non_nan_rows = ~np.isnan(X_test_processed).any(axis=1)
    X_test_final = X_test_processed[non_nan_rows]
    y_test_final = y_test.iloc[non_nan_rows]

    # --- Train SVC ---
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train_processed, y_train)
    y_pred = clf.predict(X_test_final)

    # --- Feature importance ---
    cat_feature_names = onehot.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(cat_feature_names)
    coefs = clf.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': np.abs(coefs)
    }).sort_values(by='Importance', ascending=False)

    print("\nTop Important Features (SVC):")
    print(coef_df.head(n_ftrs))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=coef_df.head(n_ftrs))
    plt.title(f"Top {n_ftrs} Feature Importances (SVC)")
    plt.tight_layout()

    file_name = f'SVC_features_top-{n_ftrs}.png'
    file_path = os.path.join("output", file_name)
    plt.savefig(file_path, dpi=300)

    #plt.show()

elif model_type == "hgb":
    # --- Categorical preprocessing ---
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = cat_encoder.fit_transform(X_train[categorical_features])
    X_test_cat = cat_encoder.transform(X_test[categorical_features])

    # Numerical features stay as-is
    X_train_num = X_train[numerical_features].to_numpy()
    X_test_num = X_test[numerical_features].to_numpy()

    # Combine - alternatively combine all numerical first and categorical second
    X_train_processed = np.hstack([X_train_cat, X_train_num])
    X_test_processed = np.hstack([X_test_cat, X_test_num])

    # Train
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train_processed, y_train)
    y_pred = clf.predict(X_test_processed)
    y_test_final = y_test

    # Feature names
    feature_names = categorical_features + numerical_features

    # SHAP
    explainer = shap.Explainer(clf)
    shap_values = explainer(X_test_processed)
    feature_importances = np.abs(shap_values.values).mean(axis=0)

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Important Features (HGB):")
    print(feature_importance_df.head(n_ftrs))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(n_ftrs), x='Importance', y='Feature')
    plt.title(f'Top {n_ftrs} Feature Importances (HGB)')
    plt.tight_layout()

    file_name = f"HGB_features_top-{n_ftrs}.png"
    file_path = os.path.join("output", file_name)
    plt.savefig(file_path, dpi=300)
    #plt.show()

else:
    raise ValueError("Invalid model_type. Choose 'svc' or 'hgb'.")

# --- Evaluation ---
label_names = ["FiErg <= 0", "FiErg > 0"]
y_test_named = [label_names[label] for label in y_test_final]
y_pred_named = [label_names[label] for label in y_pred]
accuracy = accuracy_score(y_test_final, y_pred)
print(f"\nModel: {model_type.upper()}")
print(f"Accuracy: {round(accuracy, 2)}")
print("Classification Report:")
print(classification_report(y_test_named, y_pred_named))

cm = confusion_matrix(y_test_final, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix ({model_type.upper()})')

file_name = f'confusion_matrix_{model_type}.png'
file_path = os.path.join("output", file_name)

# Save the plot to the output folder
plt.savefig(file_path, dpi=300)


#plt.show()

# ---
# import os
# import subprocess

# # === 1. Confusion Matrix Data ===

# # === 2. Convert to pandas DataFrame ===
# df_cm = pd.DataFrame(cm, index=[f"True {l}" for l in label_names],
#                         columns=[f"Pred {l}" for l in label_names])

# # === 3. Export to LaTeX table string ===
# latex_table = df_cm.to_latex(index=True, caption="Confusion Matrix", label="tab:confusion-matrix")

# # === 4. Write full .tex document ===
# tex_document = r"""
# \documentclass{article}
# \usepackage{booktabs}
# \usepackage[margin=1in]{geometry}
# \begin{document}

# """ + latex_table + r"""

# \end{document}
# """

# # === 5. Save to .tex file ===
# output_base = "confusion_matrix_table"
# tex_file = f"{output_base}.tex"
# pdf_file = f"{output_base}.pdf"

# with open(tex_file, "w") as f:
#     f.write(tex_document)
# print(f"Saved LaTeX table to: {tex_file}")




