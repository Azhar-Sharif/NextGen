import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    # Load chunks
    with open("cleaned_chunks.json", "r") as f:
        chunks = json.load(f)

    texts = [chunk['text'] for chunk in chunks]
    chunk_ids = [chunk['chunk_id'] for chunk in chunks]

    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Model is on device: {model.device}")

    # Set batch size
    batch_size = 64

    # Generate embeddings
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        with torch.autocast(device_type='mps', enabled=True):
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False
            )
        embeddings.extend(batch_embeddings)

    # Save embeddings
    embeddings = np.array(embeddings, dtype=np.float32)
    np.save("embeddings.npy", embeddings)

    # Save chunk IDs
    with open("chunk_ids.json", "w") as f:
        json.dump(chunk_ids, f)

    print(f"Successfully generated {len(chunks)} embeddings.")

if __name__ == "__main__":
    main()