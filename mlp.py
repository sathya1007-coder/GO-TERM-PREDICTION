import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score

# Define your MultiLayerPerceptron class
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.activation1(self.linear1(x))
        x = self.activation2(self.linear2(x))
        x = self.linear3(x)
        return x

# Define your dataset class
class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Define your training function
def train_model(embeddings, labels, num_classes, device, train_size=0.9, batch_size=1024, num_epochs=10):    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, train_size=train_size, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define your model
    model = MultiLayerPerceptron(input_dim=X_train.shape[1], num_classes=num_classes)
    
    # Define your optimizer, loss function, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    # Move model to appropriate device
    model.to(device)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        for embeddings, labels in tqdm(train_loader):
            embeddings, labels = embeddings.float().to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * embeddings.size(0)
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_targets.extend(labels.cpu().detach().numpy())
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_f1 = f1_score(train_targets, (train_preds > 0.5).astype(int), average='micro')
        train_f1_scores.append(train_f1)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.float().to(device), labels.float().to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * embeddings.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                val_targets.extend(labels.cpu().detach().numpy())
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_f1 = f1_score(val_targets, (val_preds > 0.5).astype(int), average='micro')
        val_f1_scores.append(val_f1)
        
        # Adjust learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train F1: {train_f1}, Val Loss: {val_loss}, Val F1: {val_f1}")

    return model, train_losses, val_losses, train_f1_scores, val_f1_scores

# Load your embeddings and labels
embeddings = np.load('bp2_bert_embeddings.npy')
labels_df = pd.read_csv('binary_vectors2.csv')
labels = labels_df.drop('EntryID', axis=1).values  # Assuming 'EntryID' is the ID column
# Set the number of classes
num_classes = 1500

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
model, train_losses, val_losses, train_f1_scores, val_f1_scores = train_model(embeddings, labels, num_classes, device)