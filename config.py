# config.py
# Models
OLLAMA_MODEL = "phi4-mini"  # Replaces MODEL_PATH
DATA_DIR = "data"
EMBEDDING_MODEL_PATH = "embedding_model" 

# Parent-Child Chunking Strategy
PARENT_CHUNK_SIZE = 800
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 150
CHILD_CHUNK_OVERLAP = 20

# Retriever
TOP_K_RETRIEVE = 10
TOP_K_PARENTS = 3
RRF_K = 60