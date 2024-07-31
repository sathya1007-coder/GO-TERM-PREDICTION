import torch
from transformers import BertTokenizer, BertModel
from Bio import SeqIO
import numpy as np

# Load the BERT-based tokenizer and model
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

# Input FASTA file containing protein sequences
fasta_file = "/content/train_sequences.fasta"

# Output file to save embeddings and accession numbers
output_file = "bert_embeddings.npy"

# Function to calculate embeddings for a sequence
def calculate_embeddings(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling over tokens
    return embeddings

# Read the FASTA file and calculate embeddings
embeddings_dict = {}
for record in SeqIO.parse(fasta_file, "fasta"):
    accession_number = record.id
    sequence = str(record.seq)

    # Calculate embeddings for the sequence
    embeddings = calculate_embeddings(sequence)

    # Store the embeddings in a dictionary
    embeddings_dict[accession_number] = embeddings.tolist()

# Save the embeddings to an npy file
np.save(output_file, embeddings_dict)

print("Embeddings saved to:", output_file)
