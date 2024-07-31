import logging
from pathlib import Path
from typing import List, Optional, Generator
from allennlp.commands.elmo import ElmoEmbedder
from numpy import ndarray
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# A random short sequence (T0922 from CASP2)
_warmup_seq = "MGSSHHHHHHSSGLVPRGSHMASVQKFPGDANCDGIVDISDAVLIMQTMANPSKYQMTDKGRINADVTGNSDGVTVLDAQFIQSYCLGLVELPPVE"


class SeqVecEmbedder(EmbedderWithFallback):
    name = "seqvec"
    embedding_dimension = 1024
    number_of_layers = 3

    _weights_file: str
    _options_file: str
    _model: ElmoEmbedder
    # The fallback model running on the CPU, which will be initialized if needed
    _model_fallback: Optional[ElmoEmbedder] = None
    necessary_files = ["weights_file", "options_file"]

    def __init__(self, warmup_rounds: int = 4, **kwargs):
        """
        Initialize Elmo embedder. Can define non-positional arguments for paths of files and other settings.

        :param warmup_rounds: A sample sequence will be embedded this often to
            work around ELMo's non-determinism.
        :param weights_file: path of weights file
        :param options_file: path of options file
        :param model_directory: Alternative of weights_file/options_file
        :param max_amino_acids: max # of amino acids to include in embed_many batches. Default: 15k AA
        """
        super().__init__(**kwargs)

        # Get file locations from kwargs
        if "model_directory" in self._options:
            self._weights_file = str(
                Path(self._options["model_directory"]).joinpath("weights_file")
            )
            self._options_file = str(
                Path(self._options["model_directory"]).joinpath("options_file")
            )
        else:
            self._weights_file = self._options["weights_file"]
            self._options_file = self._options["options_file"]

        if self._device.type == "cuda":
            logger.info("CUDA available, using the GPU")
            cuda_device = self._device.index or 0
        else:
            logger.info("CUDA NOT available, using the CPU. This is slow")
            cuda_device = -1

        self._model = ElmoEmbedder(
            weight_file=self._weights_file,
            options_file=self._options_file,
            cuda_device=cuda_device,
        )

        self.warmup_rounds = warmup_rounds
        if self.warmup_rounds > 0:
            logger.info("Running ELMo warmup")
            for _ in range(self.warmup_rounds):
                self.embed(_warmup_seq)

    def embed(self, sequence: str) -> ndarray:
        return self._model.embed_sentence(list(sequence))

    def embed_sequences_from_fasta(self, fasta_file_path: str, output_csv: str):
        # Read sequences from the FASTA file
        sequences = []
        with open(fasta_file_path, "r") as fasta_file:
            current_sequence = ""
            for line in fasta_file:
                if line.startswith(">") and current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ""
                else:
                    current_sequence += line.strip()
            if current_sequence:
                sequences.append(current_sequence)

        # Embed sequences
        embeddings = []
        for sequence in sequences:
            embeddings.append(self.embed(sequence))

        # Stack the embeddings into a single tensor
        embeddings_tensor = torch.stack(embeddings, dim=0)

        # Convert the tensor to a NumPy array
        embeddings_array = embeddings_tensor.numpy()

        # Create a DataFrame to store the embeddings
        df = pd.DataFrame(embeddings_array)

        # Save the embeddings to a CSV file
        df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    embedder = SeqVecEmbedder()

    # Provide the path to your FASTA file and the output CSV file
    fasta_file_path = "C:/Users/sathy/Desktop/cafa/train_sequences.fasta"
    output_csv = "C:/Users/sathy/Desktop/cafa/embedding1.csv"

    # Embed the sequences from the FASTA file and save the embeddings
    embedder.embed_sequences_from_fasta(fasta_file_path, output_csv)
