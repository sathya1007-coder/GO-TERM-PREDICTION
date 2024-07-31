import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Save the thresholds and process details
ensemble_details = {
    'nn_model_path': 'mlpt5best_model.h5',
    'xgb_model_path': 'best_multi_target_xgb_model.joblib',
    'nn_threshold': 0.1,  # The threshold used for the neural network predictions
    'xgb_threshold': 0.1  # The threshold used for the XGBoost predictions
}

joblib.dump(ensemble_details, 'ensemble2_details.joblib')


def ensemble_predict(X_nn, X_xgb):
    # Load ensemble details
    ensemble_details = joblib.load('ensemble_details.joblib')
    
    # Load models
    nn_model = load_model(ensemble_details['nn_model_path'])
    xgb_model = joblib.load(ensemble_details['xgb_model_path'])
    
    # Load thresholds
    nn_threshold = ensemble_details['nn_threshold']
    xgb_threshold = ensemble_details['xgb_threshold']
    
    # Predict with Neural Network
    nn_probabilities = nn_model.predict(X_nn)
    nn_predictions = (nn_probabilities >= nn_threshold).astype(int)
    
    # Predict with XGBoost
    xgb_probabilities = xgb_model.predict_proba(X_xgb)
    xgb_predictions = np.array([pred[:, 1] for pred in xgb_probabilities]).T
    xgb_predictions = (xgb_predictions >= xgb_threshold).astype(int)
    
    # Combine predictions using majority voting (consensus)
    consensus_mask = nn_predictions == xgb_predictions
    combined_predictions = np.where(consensus_mask, nn_predictions, -1)
    
    return combined_predictions, consensus_mask

# Load input data
X_nn = np.load("bp1_t5_embeddings.npy")
features_df = pd.read_csv('selected_features.csv')
labels_df = pd.read_csv('binary_vectors1.csv')

# Ensure that the labels correspond to the embeddings data
assert X_nn.shape[0] == labels_df.shape[0]

# Extract EntryID from features and labels
entryids_features = features_df['EntryID']
entryids_labels = labels_df['EntryID']

# Find the intersection of EntryIDs
common_entryids = entryids_features[entryids_features.isin(entryids_labels)].values

# Filter embeddings, features, and labels to only include common EntryIDs
indices_common_labels = labels_df['EntryID'].isin(common_entryids)
indices_common_features = features_df['EntryID'].isin(common_entryids)

X_nn_common = X_nn[indices_common_labels]
X_xgb_common = features_df.loc[indices_common_features].iloc[:, 1:].values  # Exclude EntryID column
y_true_common = labels_df.loc[indices_common_labels].iloc[:, 1:].values  # Exclude EntryID column

# Get ensemble predictions
ensemble_predictions, consensus_mask = ensemble_predict(X_nn_common, X_xgb_common)

# Evaluate the combined predictions where there is consensus
consensus_indices = consensus_mask.all(axis=1)
y_true_consensus = y_true_common[consensus_indices]
ensemble_predictions_consensus = ensemble_predictions[consensus_indices]

ensemble_precision = precision_score(y_true_consensus, ensemble_predictions_consensus, average='weighted')
ensemble_recall = recall_score(y_true_consensus, ensemble_predictions_consensus, average='weighted')
ensemble_f1 = f1_score(y_true_consensus, ensemble_predictions_consensus, average='weighted')
ensemble_accuracy_micro = accuracy_score(y_true_consensus.ravel(), ensemble_predictions_consensus.ravel())
ensemble_accuracy_macro = accuracy_score(y_true_consensus, ensemble_predictions_consensus)

print(f"Ensemble Precision: {ensemble_precision}")
print(f"Ensemble Recall: {ensemble_recall}")
print(f"Ensemble F1 Score: {ensemble_f1}")
print(f"Ensemble Micro Accuracy: {ensemble_accuracy_micro}")
print(f"Ensemble Macro Accuracy: {ensemble_accuracy_macro}")

# Save the consensus mask and predictions for future use
np.save('ensemble_predictions.npy', ensemble_predictions)
np.save('consensus_mask.npy', consensus_mask)
