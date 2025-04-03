import faiss
import numpy as np

class VectorDB:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to the index."""
        self.index.add(vectors)
    
    def search(self, query: np.ndarray, k: int = 5):
        """Search for nearest neighbors."""
        return self.index.search(query, k)
