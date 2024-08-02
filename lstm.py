import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Load embeddings and output vectors
embeddings = np.load('bp1_bert_embeddings.npy')
output_vectors_df = pd.read_csv('binary_vectors1.csv')

# Extract labels (excluding "EntryID" column)
output_vectors = output_vectors_df.iloc[:, 1:].values.astype('float64')

# Reshape embeddings to add a timestep dimension
X = embeddings.reshape(embeddings.shape[0], 1, embeddings.shape[1])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, output_vectors, test_size=0.2, random_state=42)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(output_vectors.shape[1], activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=2048, validation_data=(X_val, y_val), callbacks=[early_stopping])
