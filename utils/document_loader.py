import os
from PyPDF2 import PdfReader
from docx import Document

def load_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def load_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def load_document(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return ""

def load_documents_from_folder(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return documents
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            text = load_document(file_path)
            if text.strip():
                documents.append({"filename": filename, "content": text})
                print(f"Loaded {filename}")
    return documents