import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.load("embeddings.npy")

with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

def retrieve_top_k(query_embedding, embeddings, k = 10):
    similarities = cosine_similarity(query_embedding.reshape(1,-1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]


st.title("IR using Document Embeddings")

query = st.text_input("Enter Your Query")

def get_query_embedding(query):
    return model.encode(query)


if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)

    st.write("### Top 10 Relevant Docs")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")