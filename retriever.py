import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

class HierarchicalHybridRetriever:
    def __init__(self, model_path):
        print(f"Loading local embedding model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()
        
        self.parent_store = {}
        self.child_chunks = []
        self.bm25 = None
        self.index = None

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        faiss.normalize_L2(embedding)
        return embedding[0]

    def ingest(self, parent_store, child_chunks):
        if not child_chunks:
            return
        self.parent_store = parent_store
        self.child_chunks = child_chunks
        
        # 1. Setup Keyword Search (BM25) on Children
        tokenized_corpus = [chunk["text"].lower().split() for chunk in child_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Setup Vector Search (FAISS) on Children
        embeddings = [self.get_embedding(chunk["text"]) for chunk in child_chunks]
        embeddings = np.array(embeddings).astype('float32')
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) 
        self.index.add(embeddings)

    def retrieve(self, query, top_k_retrieve=10, top_k_parents=3, rrf_k=60):
        if not self.child_chunks:
            return []
            
        tokenized_query = query.lower().split()
        
        # 1. Get BM25 Ranks
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:top_k_retrieve]
        
        # 2. Get Vector Ranks
        query_emb = self.get_embedding(query).astype('float32').reshape(1, -1)
        distances, faiss_ranked_indices = self.index.search(query_emb, top_k_retrieve)
        faiss_ranked_indices = faiss_ranked_indices[0]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        for rank, idx in enumerate(bm25_ranked_indices):
            if bm25_scores[idx] > 0: # Only count if BM25 actually found a match
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))
                
        for rank, idx in enumerate(faiss_ranked_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Sort chunks by final RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # 4. Map back to Parents (Deduplicating along the way)
        retrieved_parents = []
        seen_parents = set()
        
        for idx in sorted_indices:
            parent_id = self.child_chunks[idx]["parent_id"]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                retrieved_parents.append(self.parent_store[parent_id])
                
            if len(retrieved_parents) >= top_k_parents:
                break
                
        return retrieved_parents