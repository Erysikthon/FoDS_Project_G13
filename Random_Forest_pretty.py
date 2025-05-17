from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc
from Data_Preparation import *

Random_Forest = RandomForestClassifier(class_weight="balanced",random_state=42)
Random_Forest.fit(X_train, y_train)

#################### Model Prediction (Base RF Model) #######################################################
y_pred = Random_Forest.predict(X_test)
y_pred_proba = Random_Forest.predict_proba(X_test)[:, 1]


#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#AUC / ROC
fp_rates, tp_rates, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fp_rates, tp_rates)
plt.figure()
plt.plot(fp_rates, tp_rates, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('output/RF_ROC.png')


#Model Evaluation
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
#Import f1 score
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

######################## Hyper Parameter Tuning ##############################################################
# n_estimator tuning
n_estimators = [50, 100, 150, 200, 250, 300]
best_roc_auc = 0
best_n_estimators = None

for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    fp_rates, tp_rates, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fp_rates, tp_rates)
    if auc_score > best_roc_auc:
        best_roc_auc = auc_score
        best_n_estimators = n
    print(f"n_estimators: {n}, AUC: {auc_score:.4f}")

print(f"Best n_estimators: {best_n_estimators}, Best AUC: {best_roc_auc:.4f}")

# Adjusted model
Random_Forest = RandomForestClassifier(n_estimators=best_n_estimators,
                                     class_weight="balanced",
                                     random_state=42)
Random_Forest.fit(X_train, y_train)

# Model Prediction
y_pred = Random_Forest.predict(X_test)
y_pred_proba = Random_Forest.predict_proba(X_test)[:, 1]


#AUC / ROC
fp_rates, tp_rates, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fp_rates, tp_rates)
plt.figure()
plt.plot(fp_rates, tp_rates, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (with hyperparameter tuning)')
plt.legend(loc='lower right')
plt.savefig('output/RF_ROC_Tuning.png')



#Most relevant features
importances = Random_Forest.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking (after Tuning):")
lists = []
for f in range(X_train.shape[1]):
    lists.append((feature_names[indices[f]], importances[indices[f]]))

#Sort the feature importances
sorted_lists = sorted(lists, key=lambda x: x[1], reverse=True)
#List to pd.DataFrame
df_importances = pd.DataFrame(sorted_lists, columns=['Feature', 'Importance'])
# Print the DataFrame
print(df_importances.head(10))

                                                                                                        #maybe feature importance plot

####################################### RF with UVFS ########################################################
UVFS_Selector = SelectKBest(score_func=f_classif, k=20) ############################## different k maybe?????????????????????
X_UVFS = UVFS_Selector.fit_transform(X_train, y_train)
X_UVFS_test = UVFS_Selector.transform(X_test)

mask = UVFS_Selector.get_support()
selected_features = X_train.columns[mask]
print("Selected features (UVFS):", selected_features)



#Optimal n_estimators
n_estimators = [5, 10, 50, 100, 105,106,107, 150]
best_roc_auc = 0
best_n_estimators = None


for stuff in n_estimators:
    rf = RandomForestClassifier(n_estimators=stuff, class_weight= "balanced", random_state=42)
    rf.fit(X_UVFS, y_train)  # Using selected features
    y_pred = rf.predict(X_UVFS_test)  # Using selected features
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_proba = rf.predict_proba(X_UVFS_test)[:, 1]  # Using selected features
    fp_rates, tp_rates, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fp_rates, tp_rates)
    if auc_score > best_roc_auc:
        best_roc_auc = auc_score
        best_n_estimators = stuff
        best_accuracy = accuracy

    print(f"n_estimators: {stuff}, AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")


print(f"Best n_estimators (UVFS): {best_n_estimators}, Best AUC: {best_roc_auc:.4f}, Accuracy: {best_accuracy:.4f}")

rf_uvfs = RandomForestClassifier(n_estimators=best_n_estimators,
                                 class_weight="balanced",
                                 random_state=42)
rf_uvfs.fit(X_UVFS, y_train)

y_pred = rf_uvfs.predict(X_UVFS_test)
y_pred_proba = rf_uvfs.predict_proba(X_UVFS_test)[:, 1]


#AUC / ROC
fp_rates, tp_rates, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fp_rates, tp_rates)
plt.figure()
plt.plot(fp_rates, tp_rates, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (UVFS)')
plt.legend(loc='lower right')
plt.savefig('output/RF_ROC_UVFS.png')

""""
#Confusion Matrix
RF_good = RandomForestClassifier(n_estimators=best_n_estimators,class_weight="balanced", random_state=42)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('output/RF_Confusion_Matrix.png')
"""



