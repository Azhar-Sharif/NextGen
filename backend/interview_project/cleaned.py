import json

def preprocess_chunks(input_file="chunks.json", output_file="cleaned_chunks.json"):
    with open(input_file, "r") as f:
        chunks = json.load(f)

    cleaned_chunks = []
    chunk_id = 0

    for chunk in chunks:
        text = chunk["text"].strip()

        # Ignore small irrelevant chunks (less than 5 words)
        if len(text.split()) < 5:
            continue  

        # Remove duplicate chunks
        if any(c["text"] == text for c in cleaned_chunks):
            continue

        # Assign unique ID
        chunk["chunk_id"] = chunk_id
        cleaned_chunks.append(chunk)
        chunk_id += 1

    # Save cleaned data
    with open(output_file, "w") as f:
        json.dump(cleaned_chunks, f, indent=2)

    print(f"✅ Cleaned {len(cleaned_chunks)} chunks and saved to {output_file}.")

# Run preprocessing
preprocess_chunks()
