import os
import json
import PyPDF2
from pdfminer.high_level import extract_text

def get_pdf_page_count(pdf_path):
    """Get the number of pages in a PDF using PyPDF2."""
    with open(pdf_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        return len(pdf.pages)

def extract_chunks_from_pdf(pdf_path):
    """Extract text from each page of the PDF and split into paragraphs."""
    # Get the book title from the filename (without extension)
    book_title = os.path.splitext(os.path.basename(pdf_path))[0]
    num_pages = get_pdf_page_count(pdf_path)
    chunks = []
    for page_num in range(num_pages):
        # Extract text from a specific page
        page_text = extract_text(pdf_path, page_numbers=[page_num])
        # Split text into paragraphs based on double newlines
        paragraphs = page_text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para:  # Only include non-empty paragraphs
                chunks.append({
                    'book': book_title,
                    'page': page_num + 1,  # Page numbers start from 1
                    'text': para
                })
    return chunks

def main():
    # Define the path to the rag-doc folder in Downloads
    books_folder = os.path.expanduser("/Users/awais/Downloads/rag-docs")
    all_chunks = []
    # Iterate through all files in the folder
    for book_file in os.listdir(books_folder):
        if book_file.endswith(".pdf"):
            pdf_path = os.path.join(books_folder, book_file)
            chunks = extract_chunks_from_pdf(pdf_path)
            all_chunks.extend(chunks)
    # Save all chunks to a JSON file
    with open("chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Extracted {len(all_chunks)} chunks from {len(os.listdir(books_folder))} books.")

if __name__ == "__main__":
    main()