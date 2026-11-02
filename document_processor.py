import os
import re
import uuid
from pypdf import PdfReader
import docx

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size, chunk_overlap):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    return chunks

def extract_text(filepath):
    text = ""
    filename = os.path.basename(filepath).lower()
    try:
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            for page in reader.pages:
                text += (page.extract_text() or "") + " "
        elif filename.endswith((".docx", ".doc")):
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        print(f"[Error] reading {filepath}: {e}")
    return clean_text(text)

def load_and_process_documents(data_dir, parent_size, parent_overlap, child_size, child_overlap):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return {}, []
    
    parent_store = {} # Stores UUID -> Parent Text
    child_chunks = [] # Stores dicts: {"text": child_text, "parent_id": uuid, "doc_name": filename}
    
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        text = extract_text(filepath)
        
        if not text:
            continue
            
        # 1. Create Parent Chunks
        parents = chunk_text(text, parent_size, parent_overlap)
        
        for parent_text in parents:
            parent_id = str(uuid.uuid4())
            # Include metadata (like filename/heading) in the parent text for the LLM
            enriched_parent = f"\n{parent_text}"
            parent_store[parent_id] = enriched_parent
            
            # 2. Create Child Chunks from this Parent
            children = chunk_text(parent_text, child_size, child_overlap)
            for child_text in children:
                # Add filename to child text so BM25 can catch exact document name searches
                child_chunks.append({
                    "text": f"{filename} - {child_text}", 
                    "parent_id": parent_id
                })
                
        print(f"Processed: {filename} ({len(parents)} parents, {len(parents) * len(children)} children)")
            
    return parent_store, child_chunks