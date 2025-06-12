"""
Support Vector classification models - evaluating different kernels (linear, polynomical and radial basis function)
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

from sklearn.feature_selection import SelectKBest, f_classif

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
    """ Combines all data preprocessing steps """
    
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Initialize transformers
        self.num_imputer = SimpleImputer(strategy = "median")
        self.scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy = "constant", fill_value = "Unknown")
        self.onehot = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)
        
        self.is_fitted = False
        
    def fit(self, X_train):
        """ Fits all transformers on training data """
        # Numerical preprocessing
        if len(self.numerical_features) > 0:
            X_train_num = self.num_imputer.fit_transform(X_train[self.numerical_features])
            self.scaler.fit(X_train_num)
        
        # Categorical preprocessing  
        if len(self.categorical_features) > 0:
            X_train_cat = self.cat_imputer.fit_transform(X_train[self.categorical_features])
            self.onehot.fit(X_train_cat)
        
        self.is_fitted = True
        return self
        
    def transform(self, X):
        """ Transforms data using fitted transformers """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        processed_parts = []
        
        # Numerical preprocessing
        if len(self.numerical_features) > 0:
            X_num = self.num_imputer.transform(X[self.numerical_features])
            X_num_scaled = self.scaler.transform(X_num)
            processed_parts.append(X_num_scaled)
        
        # Categorical preprocessing
        if len(self.categorical_features) > 0:
            X_cat = self.cat_imputer.transform(X[self.categorical_features])
            X_cat_encoded = self.onehot.transform(X_cat)
            processed_parts.append(X_cat_encoded)
        
        # Combine features
        if len(processed_parts) > 0:
            X_processed = np.hstack(processed_parts)
        else:
            X_processed = np.array([]).reshape(len(X), 0)
        
        return X_processed
    
    def fit_transform(self, X_train):
        """ Fits and transforms training data """
        return self.fit(X_train).transform(X_train)
    
    def get_feature_names(self):
        """ Gets feature names after preprocessing """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        
        feature_names = []
        
        # Add numerical feature names
        if len(self.numerical_features) > 0:
            feature_names.extend(self.numerical_features)
        
        # Add categorical feature names
        if len(self.categorical_features) > 0:
            cat_feature_names = self.onehot.get_feature_names_out(self.categorical_features)
            feature_names.extend(list(cat_feature_names))
        
        return feature_names

class FeatureSelector:
    def __init__(self, k):
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.is_fitted = False

    def fit(self, X, y):
        self.selector.fit(X, y)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted first.")
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self):
        return self.selector.get_support()

class PreprocessedSVCTrainer:
    def train_and_evaluate(self, X_train_proc, X_test_proc, y_train, y_test, kernel_type, feature_names, **kernel_params):
        # Handle NaN rows in test set
        non_nan_rows_test = ~np.isnan(X_test_proc).any(axis=1)
        non_nan_rows_train = ~np.isnan(X_train_proc).any(axis=1)
        
        X_test_final = X_test_proc[non_nan_rows_test]
        y_test_final = y_test.iloc[non_nan_rows_test] if hasattr(y_test, 'iloc') else y_test[non_nan_rows_test]
        
        X_train_final = X_train_proc[non_nan_rows_train]
        y_train_final = y_train.iloc[non_nan_rows_train] if hasattr(y_train, 'iloc') else y_train[non_nan_rows_train]
        
        # Create and train model
        if kernel_type == "linear":
            clf = SVC(kernel="linear", random_state=42, class_weight='balanced', **kernel_params)
        elif kernel_type == "poly":
            clf = SVC(kernel="poly", probability=True, **kernel_params)
        elif kernel_type == "rbf":
            clf = SVC(kernel="rbf", probability=True, **kernel_params)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        clf.fit(X_train_final, y_train_final)
        
        # Predictions
        y_pred_test = clf.predict(X_test_final)
        y_pred_train = clf.predict(X_train_final)
        
        # SHAP values - using consistent sampling
        shap_sample_size = min(100, len(X_test_final))  # Consistent sample size
        X_test_sample = X_test_final[:shap_sample_size]
        
        try:
            if kernel_type == "linear":
                explainer = shap.LinearExplainer(clf, X_train_final)
                shap_values = explainer(X_test_sample)
            else:
                # Use KernelExplainer with background sampling for non-linear kernels
                background_size = min(100, len(X_train_final))
                background = shap.sample(X_train_final, background_size, random_state=42)
                explainer = shap.KernelExplainer(clf.predict, background)
                shap_values = explainer.shap_values(X_test_sample)
        except Exception as e:
            print(f"SHAP computation failed for {kernel_type}: {e}")
            explainer = None
            shap_values = None
        
        # Get decision scores
        if hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test_final)
        else:
            y_score = clf.predict_proba(X_test_final)[:, 1]
        
        model_type = f"SVC_{kernel_type}"
        
        return {
            'X_test_processed': X_test_final,
            'X_train_processed': X_train_final,
            'X_test_sample': X_test_sample,  # Add this for SHAP plotting
            'feature_names': feature_names,
            'explainer': explainer,
            'shap_values': shap_values,
            'y_test_final': y_test_final,
            'y_train_final': y_train_final,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train,
            'y_score': y_score,
            'model_type': model_type,
            'model': clf,
        }

class PreprocessedOptunaOptimizer:
    def __init__(self, X_train_proc, y_train):
        self.X_train_processed = X_train_proc
        self.y_train = y_train
    
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
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        
        if kernel_type == "linear":
            study.optimize(self.objective_linear, n_trials=n_trials)
        elif kernel_type == "poly":
            study.optimize(self.objective_poly, n_trials=n_trials)
        elif kernel_type == "rbf":
            study.optimize(self.objective_rbf, n_trials=n_trials)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
            
        return study.best_params

def hgb_basic(X_train, X_test, numerical_features, categorical_features, y_train, y_test):
    """ HGB Basic - kept separate as it uses different preprocessing """
    model_type = "HGB"

    if len(categorical_features) > 0:
        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_cat = cat_encoder.fit_transform(X_train[categorical_features])
        X_test_cat = cat_encoder.transform(X_test[categorical_features])
    else:
        X_train_cat = np.array([]).reshape(len(X_train), 0)
        X_test_cat = np.array([]).reshape(len(X_test), 0)

    if len(numerical_features) > 0:
        X_train_num = X_train[numerical_features].to_numpy()
        X_test_num = X_test[numerical_features].to_numpy()
    else:
        X_train_num = np.array([]).reshape(len(X_train), 0)
        X_test_num = np.array([]).reshape(len(X_test), 0)
    
    # Combine features
    parts_train = [part for part in [X_train_cat, X_train_num] if part.shape[1] > 0]
    parts_test = [part for part in [X_test_cat, X_test_num] if part.shape[1] > 0]
    
    if len(parts_train) > 0:
        X_train_processed = np.hstack(parts_train)
        X_test_processed = np.hstack(parts_test)
    else:
        raise ValueError("No features available for HGB model")
    
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train_processed, y_train)
    
    y_pred_test = clf.predict(X_test_processed)
    y_pred_train = clf.predict(X_train_processed)
    
    feature_names = categorical_features + numerical_features
    
    # Consistent sampling for SHAP
    shap_sample_size = min(100, len(X_test_processed))
    X_test_sample = X_test_processed[:shap_sample_size]

    try:
        explainer = shap.Explainer(clf)
        shap_values = explainer(X_test_sample)
    except Exception as e:
        print(f"SHAP computation failed for HGB: {e}")
        explainer = None
        shap_values = None
    
    y_score = clf.predict_proba(X_test_processed)[:, 1]

    return {
        'X_test_processed': X_test_processed,
        'X_train_processed': X_train_processed,
        'X_test_sample': X_test_sample,  # Add this for SHAP plotting
        'feature_names': feature_names,
        'explainer': explainer,
        'shap_values': shap_values,
        'y_test_final': y_test,
        'y_train_final': y_train,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train,
        'y_score': y_score,
        'model_type': model_type,
    }

def plot_shap(X_test_processed, feature_names, shap_values, model_type):
    """ SHAP plotting function that handles different models and kernels """
    try:
        if hasattr(shap_values, 'data') and hasattr(shap_values, 'values'):
            shap_data = shap_values.data
            shap_vals = shap_values.values
        elif isinstance(shap_values, np.ndarray):
            n_samples = shap_values.shape[0]
            shap_data = X_test_processed[:n_samples]
            shap_vals = shap_values
        else:
            raise ValueError(f"Unexpected SHAP values format: {type(shap_values)}")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, shap_data, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot - {model_type}")
        plt.tight_layout()
        plt.savefig(f'output/{model_type}_shap_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP values plotted for {model_type}")
    except Exception as e:
        print(f"Failed to plot SHAP for {model_type}: {e}")

def plot_feature_importance(feature_names, shap_values, model_type, n_ftrs=10):
    """ Plots feature importances for different models based on shap values """
    try:
        # Handle different SHAP value formats
        if hasattr(shap_values, 'values'):
            # SHAP Explanation object
            feature_importances = np.abs(shap_values.values).mean(axis=0)
        elif isinstance(shap_values, np.ndarray):
            # Numpy array
            feature_importances = np.abs(shap_values).mean(axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values format: {type(shap_values)}")

        # Ensure we have the right number of features
        if len(feature_importances) != len(feature_names):
            print(f"Warning: Feature importance length ({len(feature_importances)}) doesn't match feature names length ({len(feature_names)})")
            min_len = min(len(feature_importances), len(feature_names))
            feature_importances = feature_importances[:min_len]
            feature_names = feature_names[:min_len]

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by="Importance", ascending=False)

        print(f"\nTop {n_ftrs} features for {model_type}:")
        print(feature_importance_df.head(n_ftrs))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df.head(n_ftrs), x='Importance', y='Feature')
        plt.title(f'Top {n_ftrs} Feature Importances - {model_type}')
        plt.tight_layout()
        plt.savefig(f'output/{model_type}_features_top-{n_ftrs}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved for {model_type}")
    except Exception as e:
        print(f"Failed to plot feature importance for {model_type}: {e}")

def evaluation(y_true, y_pred, model_type, data_type="Test"):
    """ Evaluates the prediction and draws a confusion matrix """
    label_names = ["FiErg <= 0", "FiErg > 0"]
    y_true_named = [label_names[label] for label in y_true]
    y_pred_named = [label_names[label] for label in y_pred]
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{data_type} Evaluation - Model: {model_type}")
    print(f"Accuracy: {round(accuracy, 3)}")
    print("Classification Report:")
    print(classification_report(y_true_named, y_pred_named))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_type} ({data_type})')
    plt.tight_layout()
    plt.savefig(f'output/confusion_matrix_{model_type}_{data_type.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return cm

def draw_auc(model_type, y_test_final, y_score):
    """ Draws a single AUC curve """
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
    plt.title(f'Receiver Operating Characteristic - {model_type}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'output/roc_curve_{model_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_auc_all_models(models, y_tests, y_scores):
    """ Prints all AUC curves in one graph, given a list of models """
    plt.figure(figsize=(10, 8))
    
    for n, model in enumerate(models):
        fpr, tpr, _ = roc_curve(y_tests[n], y_scores[n])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison - All Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/roc_curve_comparison_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("ROC comparison plot saved")

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
    models = []
    y_tests = []
    y_scores = []
    num_trials = 2  # reduced for faster execution - increase for better results, aim was 50, but reduced due to computational cost
    
    print(" ")
    print(f"Dataset shape: {X.shape}")
    print(f"Features: numerical={len(numerical_features)}, categorical={len(categorical_features)}")
    
    # Initialize components
    preprocessor = DataPreprocessor(numerical_features, categorical_features)

    # Preprocess the data
    print("\n Data Preprocessing: ")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Preprocessed data shape: Train={X_train_processed.shape}, Test={X_test_processed.shape}")
    
    # Apply feature selection on preprocessed data
    print(f"\nApplying feature selection (k={n_ftrs}) ")
    feature_selector = FeatureSelector(k=min(n_ftrs, X_train_processed.shape[1]))
    X_train_selected = feature_selector.fit_transform(X_train_processed, y_train)
    X_test_selected = feature_selector.transform(X_test_processed)
    
    # Get selected feature names
    feature_names = preprocessor.get_feature_names()
    selected_features_mask = feature_selector.get_support()
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]
    
    print(f"Selected {len(selected_feature_names)} features out of {len(feature_names)} total features")
    
    #print("Selected features:", selected_feature_names[:5], "..." if len(selected_feature_names) > 5 else "") -------------------------------
    print("Selected features:", selected_feature_names[:10])

    # Create trainer and optimizer for preprocessed data
    trainer = PreprocessedSVCTrainer()
    optimizer = PreprocessedOptunaOptimizer(X_train_selected, y_train)
    
    # 1. Linear SVC with optimization
    print("\n" + "="*50)
    print("Training Linear SVC")
    best_params_linear = optimizer.optimize("linear", num_trials)
    #best_params_linear = {'C': 506.158}
    print(f"Best parameters: {best_params_linear}")

    mt1 = f"SVC_linear_C-{best_params_linear['C']:.3f}"
    
    result1 = trainer.train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test, "linear", selected_feature_names, **best_params_linear)
    
    # Evaluate on both train and test sets
    evaluation(result1['y_test_final'], result1['y_pred_test'], mt1, "Test")
    evaluation(result1['y_train_final'], result1['y_pred_train'], mt1, "Train")
    draw_auc(mt1, result1['y_test_final'], result1['y_score'])
    
    if result1['shap_values'] is not None:
        plot_feature_importance(result1['feature_names'], result1['shap_values'], result1['model_type'])
        plot_shap(result1['X_test_processed'], result1['feature_names'], result1['shap_values'], result1['model_type'])
    
    models.append(mt1)
    y_tests.append(result1['y_test_final'])
    y_scores.append(result1['y_score'])
    
    # 2. Polynomial SVC with optimization  
    print("\n" + "="*50)
    print("Training Polynomial SVC")
    best_params_poly = optimizer.optimize("poly", num_trials)
    #best_params_poly = {'C': 0.177, 'degree': 5, 'gamma': 0.457}

    print(f"Best parameters: {best_params_poly}")

    mt2 = f"SVC_poly_C-{best_params_poly['C']:.3f}_D-{best_params_poly['degree']}_G-{best_params_poly['gamma']:.3f}"

    result2 = trainer.train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test, "poly", selected_feature_names, **best_params_poly)
    
    evaluation(result2['y_test_final'], result2['y_pred_test'], mt2, "Test")
    evaluation(result2['y_train_final'], result2['y_pred_train'], mt2, "Train")
    draw_auc(mt2, result2['y_test_final'], result2['y_score'])

    if result2['shap_values'] is not None:
        plot_feature_importance(result2['feature_names'], result2['shap_values'], result2['model_type'])
        plot_shap(result2['X_test_processed'], result2['feature_names'], result2['shap_values'], result2['model_type'])
    
    models.append(mt2)
    y_tests.append(result2['y_test_final'])
    y_scores.append(result2['y_score'])
    
    # 3. RBF SVC with optimization
    print("\n" + "="*50)
    print("Training RBF SVC")
    best_params_rbf = optimizer.optimize("rbf", num_trials)
    #best_params_rbf = {'C': 24.658, 'gamma': 0.098}

    print(f"Best parameters: {best_params_rbf}")

    mt3 = f"SVC_rbf_C-{best_params_rbf['C']:.3f}_G-{best_params_rbf['gamma']:.3f}"

    result3 = trainer.train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test, "rbf", selected_feature_names, **best_params_rbf)
    
    evaluation(result3['y_test_final'], result3['y_pred_test'], mt3, "Test")
    evaluation(result3['y_train_final'], result3['y_pred_train'], mt3, "Train")
    draw_auc(mt3, result3['y_test_final'], result3['y_score'])

    if result3['shap_values'] is not None:
        plot_feature_importance(result3['feature_names'], result3['shap_values'], result3['model_type'])
        plot_shap(result3['X_test_processed'], result3['feature_names'], result3['shap_values'], result3['model_type'])
    
    models.append(mt3)
    y_tests.append(result3['y_test_final'])
    y_scores.append(result3['y_score'])
    
    # 4. HGB model - not feature selected
    print("\n" + "="*50)
    print("Training Histogram Gradient Boosting")
    result4 = hgb_basic(X_train, X_test, numerical_features, categorical_features, y_train, y_test)
    
    evaluation(result4['y_test_final'], result4['y_pred_test'], result4['model_type'], "Test")
    evaluation(result4['y_train_final'], result4['y_pred_train'], result4['model_type'], "Train")
    draw_auc(result4['model_type'], result4['y_test_final'], result4['y_score'])

    if result4['shap_values'] is not None:
        plot_feature_importance(result4['feature_names'], result4['shap_values'], result4['model_type'])
        plot_shap(result4['X_test_processed'], result4['feature_names'], result4['shap_values'], result4['model_type'])
    
    models.append(result4['model_type'])
    y_tests.append(result4['y_test_final'])
    y_scores.append(result4['y_score'])
    
    # Compares of models in a single plot
    print("\n" + "="*50)
    print("Creating model comparison: ")
    draw_auc_all_models(models, y_tests, y_scores)
    
    print("\n Pipeline completed !")
    print("\n Generated files: ")
    print("Individual confusion matrices for each model (train/test)")
    print("Individual ROC curves for each model")
    print("SHAP plots and feature importance plots")
    print("Combined ROC curve comparison")

if __name__ == "__main__":
    main()
