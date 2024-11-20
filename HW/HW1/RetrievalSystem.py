from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine
import numpy as np
from nltk.tokenize import word_tokenize

class RetrievalSystem:
    def __init__(self, documents, queries, fasttext_models):
        """
        Initializes the Retrieval System.

        Parameters:
        - documents: DataFrame with columns ['doc_id', 'processed_text'].
        - queries: DataFrame with columns ['query_id', 'processed_text'].
        - fasttext_models: Dictionary with keys 'scratch', 'pretrained', and 'finetuned' pointing to respective FastText models.
        """
        self.documents = documents
        self.queries = queries
        self.fasttext_models = fasttext_models
        
        # BM25 setup
        self.bm25 = BM25Okapi([doc.split() for doc in documents['processed_text']])

        # TF-IDF setup
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents['processed_text'])

        # Precompute document embeddings for each FastText model
        self.document_embeddings = {}
        for model_key, model in fasttext_models.items():
            self.document_embeddings[model_key] = [
                self.get_sentence_embedding(doc, model) for doc in documents['processed_text']
            ]

    def get_sentence_embedding(self, sentence, model):
        """
        Computes the average embedding of a sentence using a FastText model.

        Parameters:
        - sentence: Input text (str).
        - model: FastText model.

        Returns:
        - Average embedding vector (numpy array).
        """
        # Tokenize the input sentence
        words = word_tokenize(sentence) if isinstance(sentence, str) else []

        # Retrieve embeddings for words in the model's vocabulary
        word_vectors = [model[word] for word in words if word in model]

        if not word_vectors:
            # If no words are found in the model, return a zero vector
            return np.zeros(model.get_dimension())
        
        # Compute the average of the embeddings
        return np.mean(word_vectors, axis=0)
    
    def get_embedding(self, text, model_key):
        """
        Gets the embedding of a sentence using the selected FastText model.

        Parameters:
        - text: Input text (str).
        - model_key: Key of the FastText model ('scratch', 'pretrained', 'finetuned').

        Returns:
        - Embedding vector (numpy array).
        """
        if model_key not in self.fasttext_models:
            raise ValueError(f"Invalid model key '{model_key}'. Expected one of: {list(self.fasttext_models.keys())}")
        model = self.fasttext_models[model_key]
        return self.get_sentence_embedding(text, model)

    def retrieve(self, query, tfidf_weight=0.4, bm25_weight=0.4, embedding_weight=0.2, fasttext_model_key='finetuned', top_k=10):
        """
        Retrieves top documents for a given query.

        Parameters:
        - query: Input query text (str).
        - tfidf_weight, bm25_weight, embedding_weight: Weights for TF-IDF, BM25, and embeddings.
        - fasttext_model_key: Key of the FastText model ('scratch', 'pretrained', 'finetuned').
        - top_k: Number of top documents to retrieve.

        Returns:
        - DataFrame with top-k documents and their scores.
        """
        # Compute query vector for TF-IDF
        query_vector = self.tfidf_vectorizer.transform([query])

        # Compute TF-IDF scores
        tfidf_scores = np.array(query_vector.dot(self.tfidf_matrix.T).toarray()[0])

        # Compute BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query.split()))

        # Compute embedding scores
        query_embedding = self.get_embedding(query, fasttext_model_key)
        doc_embeddings = self.document_embeddings[fasttext_model_key]
        embedding_scores = np.array([
            1 - cosine(query_embedding, doc_emb) if np.any(doc_emb) else 0
            for doc_emb in doc_embeddings
        ])

        # Normalize scores
        tfidf_scores = tfidf_scores / np.max(tfidf_scores) if np.max(tfidf_scores) > 0 else tfidf_scores
        bm25_scores = bm25_scores / np.max(bm25_scores) if np.max(bm25_scores) > 0 else bm25_scores
        embedding_scores = embedding_scores / np.max(embedding_scores) if np.max(embedding_scores) > 0 else embedding_scores

        # Compute final scores
        combined_scores = (
            tfidf_weight * tfidf_scores +
            bm25_weight * bm25_scores +
            embedding_weight * embedding_scores
        )

        # Retrieve top-k documents
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return self.documents.iloc[top_indices], combined_scores[top_indices]
