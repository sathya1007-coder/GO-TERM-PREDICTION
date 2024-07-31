import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

# Step 1: Load the Data
labels_file = "binary_vectors1.csv"
features_file = "selected_features.csv"
labels_df = pd.read_csv(labels_file)
features_df = pd.read_csv(features_file)

# Step 2: Preprocess the Data
labels_df.set_index('EntryID', inplace=True)
features_df.set_index('EntryID', inplace=True)

common_entry_ids = labels_df.index.intersection(features_df.index)
labels_df = labels_df.loc[common_entry_ids]
features_df = features_df.loc[common_entry_ids]

# Step 3: Split Data and Train the XGBoost Model
X_train, X_val, y_train, y_val = train_test_split(features_df, labels_df, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()

# Define cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=cv, scoring='accuracy')

# Train model on the entire training set
model.fit(X_train, y_train)

# Find the best model based on cross-validation accuracy
best_cv_score = max(cv_scores)
best_cv_model_index = list(cv_scores).index(best_cv_score)

# Retrain the model with the entire training data
model.fit(X_train, y_train)

# Step 4: Evaluate the Model on Validation Data
val_predictions = model.predict(X_val)

# Calculate accuracy for validation data
val_accuracy = accuracy_score(y_val, val_predictions)

# Print the validation accuracy
print("Validation Accuracy:", val_accuracy)

# Step 5: Save the Best Model
model.save_model("best_xgboost_model.bin")
