from sklearn.ensemble import RandomForestClassifier

Random_Forest = RandomForestClassifier(class_weight="balanced",random_state=42)
Random_Forest.fit(X_train, y_train)

#Model Prediction
y_pred = Random_Forest.predict(X_test)
y_pred_proba = Random_Forest.predict_proba(X_test)[:, 1]


#Confusion Matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('Confusion.png')
