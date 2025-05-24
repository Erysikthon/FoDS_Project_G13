"""

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.svm import LinearSVC
import shap
import optuna
from Data_Preparation import df

tt_size = 0.2

n_ftrs = 10

data = df.copy()

label = "FiErg"
features = ["KT", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]

data = data.dropna(subset=[label])
columns_to_drop = [col for col in features if data[col].isna().sum() == len(data[col])]

data.drop(columns = columns_to_drop, inplace=True)
features = [col for col in features if col not in columns_to_drop]

# Rename columns
if "AnzBelP (nur ab KZP2010)" in data.columns:
    data.rename(columns={"AnzBelP (nur ab KZP2010)": "AnzBelP"}, inplace=True)
    features = ["AnzBelP" if col == "AnzBelP (nur ab KZP2010)" else col for col in features]

data[label] = (data[label] > 0).astype(int)

X = data[features]
y = data[label]
yar = data["JAHR"]

stratify_key = pd.Series(list(zip(y, yar)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tt_size, random_state=42, stratify = stratify_key)

# Feature types
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()



class DataPreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Initialize transformers
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="NA")
        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        self.is_fitted = False
        
    def fit(self, X_train):
        """Fit all transformers on training data"""
        # Numerical preprocessing
        X_train_num = self.num_imputer.fit_transform(X_train[self.numerical_features])
        self.scaler.fit(X_train_num)
        
        # Categorical preprocessing  
        X_train_cat = self.cat_imputer.fit_transform(X_train[self.categorical_features])
        self.onehot.fit(X_train_cat)
        
        self.is_fitted = True
        return self
        
    def transform(self, X):
        """Transform data using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        # Numerical preprocessing
        X_num = self.num_imputer.transform(X[self.numerical_features])
        X_num_scaled = self.scaler.transform(X_num)
        
        # Categorical preprocessing
        X_cat = self.cat_imputer.transform(X[self.categorical_features])
        X_cat_encoded = self.onehot.transform(X_cat)
        
        # Combine features
        X_processed = np.hstack([X_num_scaled, X_cat_encoded])
        
        return X_processed
    
    def fit_transform(self, X_train):
        """Fit and transform training data"""
        return self.fit(X_train).transform(X_train)
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        cat_feature_names = self.onehot.get_feature_names_out(self.categorical_features)
        return self.numerical_features + list(cat_feature_names)

class SVCModelTrainer:
    """Unified SVC model trainer for different kernels"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, kernel_type, **kernel_params):
        """Train SVC model with specified kernel and parameters"""
        
        # Preprocess data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Handle NaN rows in test set
        non_nan_rows = ~np.isnan(X_test_processed).any(axis=1)
        X_test_final = X_test_processed[non_nan_rows]
        y_test_final = y_test.iloc[non_nan_rows]
        
        # Create and train model
        if kernel_type == "linear":
            clf = SVC(kernel="linear", random_state=42, class_weight='balanced', **kernel_params)
        elif kernel_type == "poly":
            clf = SVC(kernel="poly", probability=True, **kernel_params)
        elif kernel_type == "rbf":
            clf = SVC(kernel="rbf", probability=True, **kernel_params)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        clf.fit(X_train_processed, y_train)
        y_pred = clf.predict(X_test_final)
        
        # Get feature names
        feature_names = self.preprocessor.get_feature_names()
        
        # SHAP values (only for linear kernel)
        if kernel_type == "linear":
            explainer = shap.LinearExplainer(clf, X_train_processed)
            shap_values = explainer(X_test_processed)
        else:
            explainer = None # due to time consuming activity
            shap_values = None
        
        # Get decision scores
        y_score = clf.decision_function(X_test_final)
        
        model_type = f"SVC_{kernel_type}"
        
        return {
            'X_test_processed': X_test_processed,
            'feature_names': feature_names,
            'explainer': explainer,
            'shap_values': shap_values,
            'y_test_final': y_test_final,
            'y_pred': y_pred,
            'y_score': y_score,
            'model_type': model_type,
            'model': clf, 
        }


def hgb_basic(X_train, X_test, numerical_features, categorical_features, y_train, y_test):
    """HGB Basic - kept separate as it uses different preprocessing"""
    model_type = "HGB"

    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = cat_encoder.fit_transform(X_train[categorical_features])
    X_test_cat = cat_encoder.transform(X_test[categorical_features])

    X_train_num = X_train[numerical_features].to_numpy()
    X_test_num = X_test[numerical_features].to_numpy()
    X_train_processed = np.hstack([X_train_cat, X_train_num])
    X_test_processed = np.hstack([X_test_cat, X_test_num])
    
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train_processed, y_train)
    y_pred = clf.predict(X_test_processed)
    y_test_final = y_test

    feature_names = categorical_features + numerical_features

    explainer = shap.Explainer(clf)
    shap_values = explainer(X_test_processed)
    y_score = clf.predict_proba(X_test_processed)[:, 1]

    return {
        'X_test_processed': X_test_processed,
        'feature_names': feature_names,
        'explainer': explainer,
        'shap_values': shap_values,
        'y_test_final': y_test_final,
        'y_pred': y_pred,
        'y_score': y_score,
        'model_type': model_type,
    }


def plot_shap(X_test_processed, feature_names, shap_values, model_type):
    plt.figure()
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
    plt.title(f"\n{model_type}")
    plt.savefig(f'{model_type}_shap_values.png', dpi=300)
    plt.close()
    print("SHAP values plotted")


def plot_feature_importance(feature_names, shap_values, model_type, n_ftrs=10):
    """Plot shap value based feature importance"""
    feature_importances = np.abs(shap_values.values).mean(axis=0)

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by="Importance", ascending=False)

    print(feature_importance_df.head(n_ftrs))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(n_ftrs), x='Importance', y='Feature')
    plt.title(f'Top {n_ftrs} Feature Importances {model_type}')
    plt.tight_layout()
    plt.savefig(f'{model_type}_features_top-{n_ftrs}.png', dpi=300)


def evaluation(y_test_final, y_pred, model_type):
    """Evaluates the prediction and draws a confusion matrix"""
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
    plt.savefig(f'confusion_matrix_{model_type}.png', dpi=300)

    return cm


def draw_auc(model_type, y_test_final, y_score):
    """Draws a single AUC curve"""
    fpr, tpr, thresholds_ = roc_curve(y_test_final, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.3f}")

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = {:.3f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic ({model_type.upper()})')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_type}.png', dpi=300)


def draw_auc_all_models(models, y_tests, y_scores):
    """Given a list of models, prints all AUC curves in one graph"""
    plt.figure()
    
    for n, model in enumerate(models):
        fpr, tpr, _ = roc_curve(y_tests[n], y_scores[n])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve {model} (area = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC comparison')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve_comparison_all_models.png', dpi=300)


class OptunaOptimizer:
    """Handles Optuna optimization for different SVC kernels"""
    
    def __init__(self, X_train, y_train, numerical_features, categorical_features):
        self.X_train = X_train
        self.y_train = y_train
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Preprocess data once for optimization
        self.preprocessor = DataPreprocessor(numerical_features, categorical_features)
        self.X_train_processed = self.preprocessor.fit_transform(X_train)
    
    def objective_linear(self, trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        clf = SVC(kernel="linear", C=C, random_state=42, class_weight='balanced')
        scores = cross_val_score(clf, self.X_train_processed, self.y_train, cv=3, scoring="roc_auc")
        return scores.mean()

    def objective_poly(self, trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        degree = trial.suggest_int('degree', 2, 5)
        gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
        
        clf = SVC(kernel="poly", C=C, degree=degree, gamma=gamma, probability=True)
        scores = cross_val_score(clf, self.X_train_processed, self.y_train, cv=3, scoring="roc_auc")
        return scores.mean()

    def objective_rbf(self, trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
        
        clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True)
        scores = cross_val_score(clf, self.X_train_processed, self.y_train, cv=3, scoring="roc_auc")
        return scores.mean()

    def optimize(self, kernel_type, n_trials=10):
        """Optimize hyperparameters for specified kernel"""
        study = optuna.create_study(direction="maximize")
        
        if kernel_type == "linear":
            study.optimize(self.objective_linear, n_trials=n_trials)
        elif kernel_type == "poly":
            study.optimize(self.objective_poly, n_trials=n_trials)
        elif kernel_type == "rbf":
            study.optimize(self.objective_rbf, n_trials=n_trials)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
            
        return study.best_params


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    models = []
    y_tests = []
    y_scores = []
    num_trials = 30 # change to at least 10
    
    # Initialize components
    preprocessor = DataPreprocessor(numerical_features, categorical_features)

    trainer = SVCModelTrainer(preprocessor)

    # X_train = preprocessor.fit_transform(X_train, y_train)
    # X_test = preprocessor.transform(X_test)


    optimizer = OptunaOptimizer(X_train, y_train, numerical_features, categorical_features)
    
    # 1. Linear SVC with optimization
    best_params_linear = optimizer.optimize("linear", num_trials)
    mt1 = f"SVC_best_linear_C-{best_params_linear['C']:.2f}"
    
    result1 = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, "linear", **best_params_linear)
    evaluation(result1['y_test_final'], result1['y_pred'], mt1)
    
    if result1['shap_values'] is not None:
        plot_feature_importance(result1['feature_names'], result1['shap_values'], result1['model_type'])
        plot_shap(result1['X_test_processed'], result1['feature_names'], result1['shap_values'], result1['model_type'])
    
    models.append(mt1)
    y_tests.append(result1['y_test_final'])
    y_scores.append(result1['y_score'])
    
    # 2. Polynomial SVC with optimization
    best_params_poly = optimizer.optimize("poly", num_trials)
    mt2 = f"SVC_poly_C-{best_params_poly['C']:.2f}_D-{best_params_poly['degree']:.2f}_G-{best_params_poly['gamma']:.2f}"
    
    result2 = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, "poly", **best_params_poly)
    evaluation(result2['y_test_final'], result2['y_pred'], mt2)
    
    models.append(mt2)
    y_tests.append(result2['y_test_final'])
    y_scores.append(result2['y_score'])
    
    # 3. RBF SVC with optimization
    best_params_rbf = optimizer.optimize("rbf", num_trials)
    mt3 = f"SVC_rbf_C-{best_params_rbf['C']:.2f}_G-{best_params_rbf['gamma']:.2f}"
    
    result3 = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, "rbf", **best_params_rbf)
    evaluation(result3['y_test_final'], result3['y_pred'], mt3)
    
    models.append(mt3)
    y_tests.append(result3['y_test_final'])
    y_scores.append(result3['y_score'])
    
    # 4. HGB model
    result4 = hgb_basic(X_train, X_test, numerical_features, categorical_features, y_train, y_test)
    evaluation(result4['y_test_final'], result4['y_pred'], result4['model_type'])
    
    models.append(result4['model_type'])
    y_tests.append(result4['y_test_final'])
    y_scores.append(result4['y_score'])
    
    # Draw comparison of all models
    draw_auc_all_models(models, y_tests, y_scores)


if __name__ == "__main__":
    main()
