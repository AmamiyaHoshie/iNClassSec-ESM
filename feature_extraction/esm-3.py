import torch
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_model_tokenizers
from Bio import SeqIO
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ESM3_sm_open_v0().to(device)
model.eval()

tokenizers = get_model_tokenizers()
sequence_tokenizer = tokenizers.sequence

fasta_file = "dataset/PeNGaRoo_train_P.fasta.txt"

embeddings_list = []

for record in SeqIO.parse(fasta_file, "fasta"):
    sequence = str(record.seq)
    print(f"Processing sequence of length {len(sequence)}")

    tokens = sequence_tokenizer.encode(sequence)
    sequence_tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sequence_tokens=sequence_tokens)
        embeddings = output.embeddings  # [batch_size, seq_len, embedding_dim]

    seq_len = sequence_tokens.size(1)
    embeddings = embeddings[:, 1:seq_len - 1, :]  # [batch_size, seq_len - 2, embedding_dim]

    feature_vector = embeddings.mean(dim=1)  # [batch_size, embedding_dim]

    feature_vector_np = feature_vector.cpu().numpy()
    embeddings_list.append(feature_vector_np)

output_file = "features/esm-3/PeNGaRoo_train_P.npy"
np.save(output_file, np.vstack(embeddings_list))

print(f"Embeddings have been saved to {output_file}")
