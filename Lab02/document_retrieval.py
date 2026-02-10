"""
Information Retrieval Task: Build a Document Retrieval System using Word2Vec

Task: Given a query string, find the most relevant documents from the Reuters corpus 
using Word2Vec embeddings.

Steps:
1. Preprocess the query string by tokenizing and removing stop words
2. Compute the average Word2Vec embedding for the query string
3. Compute the average Word2Vec embedding for each document in the Reuters corpus
4. Use cosine similarity to find the top N most relevant documents for the query
5. Display the top N document IDs and their similarity scores
"""

import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('reuters', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text, stop_words):
    """
    Preprocess text by tokenizing and removing stop words
    
    Args:
        text: Raw text string
        stop_words: Set of stop words to remove
    
    Returns:
        List of preprocessed tokens
    """
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stop words and non-alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return tokens

def get_document_embedding(tokens, model):
    """
    Compute average Word2Vec embedding for a document
    
    Args:
        tokens: List of tokens
        model: Trained Word2Vec model
    
    Returns:
        Average embedding vector
    """
    word_vectors = []
    
    for token in tokens:
        if token in model.wv:
            word_vectors.append(model.wv[token])
    
    if len(word_vectors) == 0:
        # Return zero vector if no words found in vocabulary
        return np.zeros(model.vector_size)
    
    # Return average of all word vectors
    return np.mean(word_vectors, axis=0)

def build_retrieval_system():
    """
    Build the document retrieval system
    
    Returns:
        model: Trained Word2Vec model
        doc_embeddings: Document embeddings matrix
        doc_ids: List of document IDs
        stop_words: Set of stop words
    """
    print("Building Document Retrieval System...")
    print("=" * 50)
    
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # Get all document IDs
    doc_ids = reuters.fileids()
    print(f"Total documents in corpus: {len(doc_ids)}")
    
    # Preprocess all documents
    print("Preprocessing documents...")
    corpus_sentences = []
    
    for doc_id in doc_ids:
        raw_text = reuters.raw(doc_id)
        tokens = preprocess_text(raw_text, stop_words)
        corpus_sentences.append(tokens)
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=corpus_sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=10
    )
    
    print(f"Vocabulary size: {len(model.wv)}")
    
    # Compute document embeddings
    print("Computing document embeddings...")
    doc_embeddings = []
    
    for tokens in corpus_sentences:
        embedding = get_document_embedding(tokens, model)
        doc_embeddings.append(embedding)
    
    doc_embeddings = np.array(doc_embeddings)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    print("=" * 50)
    
    return model, doc_embeddings, doc_ids, stop_words

def retrieve_documents(query, model, doc_embeddings, doc_ids, stop_words, top_n=5):
    """
    Retrieve top N most relevant documents for a query
    
    Args:
        query: Query string
        model: Trained Word2Vec model
        doc_embeddings: Document embeddings matrix
        doc_ids: List of document IDs
        stop_words: Set of stop words
        top_n: Number of top documents to return
    
    Returns:
        List of (doc_id, similarity_score, preview) tuples
    """
    # Step 1: Preprocess query
    query_tokens = preprocess_text(query, stop_words)
    
    if len(query_tokens) == 0:
        print("Warning: No valid tokens in query after preprocessing!")
        return []
    
    # Step 2: Compute query embedding
    query_embedding = get_document_embedding(query_tokens, model)
    
    if np.all(query_embedding == 0):
        print("Warning: Query embedding is zero vector!")
        return []
    
    query_embedding = query_embedding.reshape(1, -1)
    
    # Step 3 & 4: Compute cosine similarity with all documents
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get top N indices
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Prepare results
    results = []
    for idx in top_indices:
        doc_id = doc_ids[idx]
        similarity = similarities[idx]
        
        # Get document preview
        raw_text = reuters.raw(doc_id)
        # Get first 100 characters for preview
        preview = ' '.join(raw_text.split()[:15]) + '...'
        
        results.append((doc_id, similarity, preview))
    
    return results

def display_results(query, results):
    """
    Display retrieval results in the required format
    
    Args:
        query: Query string
        results: List of (doc_id, similarity_score, preview) tuples
    """
    print("\n" + "=" * 80)
    print("Top Relevant Documents:")
    print("=" * 80)
    
    for doc_id, similarity, preview in results:
        print(f"Document ID: {doc_id}, Similarity Score: {similarity:.4f}")
        print(f"Document Content: {preview}")
        print()

def main():
    """
    Main function to demonstrate the document retrieval system
    """
    # Build the retrieval system
    model, doc_embeddings, doc_ids, stop_words = build_retrieval_system()
    
    # Example queries
    queries = [
        "international trade agreement buffer stock",
        "oil prices crude production",
        "economic growth inflation"
    ]
    
    print("\n" + "=" * 80)
    print("DOCUMENT RETRIEVAL EXAMPLES")
    print("=" * 80)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retrieve_documents(query, model, doc_embeddings, doc_ids, stop_words, top_n=5)
        display_results(query, results)

if __name__ == "__main__":
    main()
