import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Load input data (change 'input_file.csv' to your actual file path)
df = pd.read_csv('input_file.csv')

# Group by EntryID and aggregate terms
grouped_df = df.groupby('EntryID').agg({
    'term': lambda x: list(x),
    'aspect': lambda x: list(set(x))  # assuming a protein can have multiple aspects
}).reset_index()

# Count of terms
grouped_df['count'] = grouped_df['term'].apply(len)

# Save the count and aspect information
count_aspect_df = grouped_df[['EntryID', 'count', 'aspect']]
count_aspect_df.to_csv('count_aspect.csv', index=False)

# Extract unique terms
unique_terms = list(set([item for sublist in grouped_df['term'] for item in sublist]))

# Create a MultiLabelBinarizer object
mlb = MultiLabelBinarizer(classes=unique_terms)
mlb.fit([unique_terms])

# Transform the list of terms for each EntryID into binary vectors
binary_vectors = mlb.transform(grouped_df['term'])

# Create a DataFrame with binary vectors
binary_vectors_df = pd.DataFrame(binary_vectors, columns=mlb.classes_)

# Concatenate EntryID column with binary vectors DataFrame
binary_vectors_df.insert(0, 'EntryID', grouped_df['EntryID'])

# Write the binary vectors DataFrame to a CSV file
binary_vectors_df.to_csv('binary_vectors1.csv', index=False)
