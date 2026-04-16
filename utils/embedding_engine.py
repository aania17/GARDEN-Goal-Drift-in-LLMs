from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, text):
        return self.model.encode(text)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def drift_score(self, goal_vec, step_vec):
        similarity = self.cosine_similarity(goal_vec, step_vec)
        return 1 - similarity