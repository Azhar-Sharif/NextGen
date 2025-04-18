import faiss
import numpy as np
import json

def build_faiss_index(embeddings_path, index_path):
    """
    Builds a FAISS index from the embeddings and saves it to a file.
    
    Args:
        embeddings_path (str): Path to the .npy file containing the embeddings.
        index_path (str): Path where the FAISS index will be saved.
    """
    # Load embeddings
    embeddings = np.load(embeddings_path).astype('float32')
    # Get the dimension of the embeddings
    d = embeddings.shape[1]
    # Create a flat L2 index
    index = faiss.IndexFlatL2(d)
    # Add embeddings to the index
    index.add(embeddings)
    # Save the index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

def verify_chunks_and_index(chunks_path, index_path):
    """
    Verifies that the number of chunks matches the number of vectors in the FAISS index.
    
    Args:
        chunks_path (str): Path to the JSON file containing the chunks.
        index_path (str): Path to the FAISS index file.
    """
    # Load chunks
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    # Load FAISS index
    index = faiss.read_index(index_path)
    # Check if the number of chunks matches the number of vectors in the index
    assert len(chunks) == index.ntotal, "Mismatch between number of chunks and index size"
    print("Verification successful: Number of chunks matches the index size.")

if __name__ == "__main__":
    # Paths to your files
    embeddings_path = 'embeddings.npy'
    chunks_path = 'cleaned_chunks.json'
    index_path = 'faiss_index.bin'
    
    # Build and save the FAISS index
    build_faiss_index(embeddings_path, index_path)
    
    # Verify that the chunks and index are consistent
    verify_chunks_and_index(chunks_path, index_path)