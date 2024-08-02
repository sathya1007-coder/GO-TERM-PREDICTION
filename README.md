# GO-TERM-PREDICTION

****Overview:****

This repository contains the code and resources for predicting protein functions using advanced deep learning techniques and computational methods. Our approach addresses the limitations of current structure prediction methods by leveraging machine learning models and feature engineering to accurately annotate protein functions based on the Gene Ontology classification.

![inbox_1313949_f9726a776920fc16813d021068259a8e_ex (3)](https://github.com/user-attachments/assets/fc1857e9-06aa-40dc-9557-2dea3144a74f)


****Contents:****

**1. ProtBert_Embeddings.py:** Generates embeddings for protein sequences using the ProtBERT model.

**2. protT5_Embeddings.py:** Generates embeddings using the ProtT5 model.

**3. seqvec_embedder.py:** Produces embeddings from protein sequences using SeqVec.

**4. cvxg.py:** Implements Cross-Validation with XGBoost for model training.

**5. mlp.py:** Contains code for building a cross-validation model using a Multilayer Perceptron.

**6. lstm.py:** Contains code for building a cross-validation model using a Long Short-Term Memory (LSTM) model.

**7. ensemble2.py:** Combines predictions from multiple models to create an ensemble model for improved accuracy.

The tabular features are extracted using the **iFeature package [https://doi.org/10.1093/nar/gkac351](url). GitHub link: [https://github.com/Superzchen/iFeature](url).**



****Usage:****

**1. Data Extraction:** Extract multiple Gene Ontology (GO) terms for each protein from the GOA database. Organize them by associating each protein ID with its corresponding list of GO terms.

**2. Label Preparation:** create a file which has proteins and their list of associated GO terms. Now, Using the **output_vector.py** file, each term was transformed into a separate column, while proteins were represented as rows in a binary matrix. Each term was encoded as either 0 or 1, with 0 indicating the absence and 1 indicating the presence of each GO term for each protein. 

**3. Model Execution:** The models can be trained and evaluated on each GO aspect (e.g., molecular function, biological process) individually and separately.

**4. Environment Setup:** The codes are designed to run on Google Cloud TPU machines. For optimal performance, use at least Kaggle's TPU resources. Ensure a proper Python environment with the required dependencies installed.


****Getting Started****

**Clone the repository:**

git clone:** [https://github.com/sathya1007-coder/GO-TERM-PREDICTION.git](url)**

**cd protein-function-prediction**

**Install the necessary dependencies:**

**pip install -r requirements.txt**

Run the feature extraction scripts to generate embeddings from protein sequences.

Train the machine learning models using the provided training scripts.

Evaluate the model performance using the evaluation scripts.

Author:

**Sathya Narayanan - Bioinformatician
**

Contact:

For any inquiries or collaboration opportunities, please reach out to me via officemailsathya@gmail.com.
