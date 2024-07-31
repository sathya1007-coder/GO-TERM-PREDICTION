import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
import joblib
import numpy as np

# Load feature and label data from CSV files
features_df = pd.read_csv('features.csv')  # Assuming features.csv has columns: EntryID, feature1, feature2, ..., feature50
labels_df = pd.read_csv('labels.csv')  # Assuming labels.csv has columns: EntryID, GO:0000002, GO:0000003, ..., GO:XXXXXXX

# Ensure label data contains only the EntryIDs present in the feature data
labels_df = labels_df[labels_df['EntryID'].isin(features_df['EntryID'])]

# Merge the feature and label data on EntryID
merged_df = features_df.merge(labels_df, on='EntryID')

# Identify feature and label columns
feature_columns = features_df.columns[1:]  # All columns except 'EntryID'
label_columns = labels_df.columns[1:]  # All columns except 'EntryID'

# Split merged data into features and labels
X = merged_df[feature_columns]
y = merged_df[label_columns]

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_cv_score = -np.inf
best_fold = None

# Store metrics for each fold
cv_metrics = []

for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize XGBoost Classifier
    xgb = XGBClassifier(random_state=42)
    multi_target_xgb = MultiOutputClassifier(xgb, n_jobs=-1)

    # Fit the model
    multi_target_xgb.fit(X_train, y_train)
    y_val_pred = multi_target_xgb.predict(X_val)

    # Calculate metrics
    fold_precision = precision_score(y_val, y_val_pred, average='weighted')
    fold_recall = recall_score(y_val, y_val_pred, average='weighted')
    fold_f1 = 2 * (fold_precision * fold_recall) / (fold_precision + fold_recall)
    
    fold_accuracy_micro = accuracy_score(y_val.values.ravel(), y_val_pred.ravel())
    fold_accuracy_macro = accuracy_score(y_val, y_val_pred)
    
    # Overall CV accuracy
    fold_cv_accuracy = (y_val_pred == y_val).mean().mean()

    cv_metrics.append({
        'fold': fold,
        'precision': fold_precision,
        'recall': fold_recall,
        'f1_score': fold_f1,
        'micro_accuracy': fold_accuracy_micro,
        'macro_accuracy': fold_accuracy_macro,
        'cv_accuracy': fold_cv_accuracy
    })

    if fold_cv_accuracy > best_cv_score:
        best_cv_score = fold_cv_accuracy
        best_model = multi_target_xgb
        best_fold = fold

    print(f"Fold {fold} - Precision: {fold_precision}, Recall: {fold_recall}, F1 Score: {fold_f1}, Micro Accuracy: {fold_accuracy_micro}, Macro Accuracy: {fold_accuracy_macro}, CV Accuracy: {fold_cv_accuracy}")

# Save the best model with compression
joblib.dump(best_model, 'best_multi_target_xgb_model.joblib', compress=5)

# Save CV metrics to a file
cv_metrics_df = pd.DataFrame(cv_metrics)
cv_metrics_df.to_csv('cv_metrics.txt', sep='\t', index=False)

# Print overall CV metrics
print("Cross-validation metrics for each fold:")
print(cv_metrics_df)

# Overall metrics
overall_precision = cv_metrics_df['precision'].mean()
overall_recall = cv_metrics_df['recall'].mean()
overall_f1_score = cv_metrics_df['f1_score'].mean()
overall_micro_accuracy = cv_metrics_df['micro_accuracy'].mean()
overall_macro_accuracy = cv_metrics_df['macro_accuracy'].mean()
overall_cv_accuracy = cv_metrics_df['cv_accuracy'].mean()

print(f"Overall Precision (CV): {overall_precision}")
print(f"Overall Recall (CV): {overall_recall}")
print(f"Overall F1 Score (CV): {overall_f1_score}")
print(f"Overall Micro Accuracy (CV): {overall_micro_accuracy}")
print(f"Overall Macro Accuracy (CV): {overall_macro_accuracy}")
print(f"Overall CV Accuracy: {overall_cv_accuracy}")

with open('cv_overall_metrics.txt', 'w') as f:
    f.write(f"Best Fold: {best_fold}\n")
    f.write(f"Overall Precision (CV): {overall_precision}\n")
    f.write(f"Overall Recall (CV): {overall_recall}\n")
    f.write(f"Overall F1 Score (CV): {overall_f1_score}\n")
    f.write(f"Overall Micro Accuracy (CV): {overall_micro_accuracy}\n")
    f.write(f"Overall Macro Accuracy (CV): {overall_macro_accuracy}\n")
    f.write(f"Overall CV Accuracy: {overall_cv_accuracy}\n")
