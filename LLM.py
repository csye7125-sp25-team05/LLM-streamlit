#!/usr/bin/env python3
import os
import streamlit as st
from pinecone import Pinecone  # Updated import
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv(".env")  # Load environment variables from .env file
# ------------------------------------------------------------------------------
# Initialization: Configure Pinecone and Gemini using environment variables
# ------------------------------------------------------------------------------
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV       = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "pdf-embeddings")
GENIE_API_KEY      = os.getenv("GENIE_API_KEY")

# Updated Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

genai.configure(api_key=GENIE_API_KEY)
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL  = "gemini-1.5-pro-latest"

# ------------------------------------------------------------------------------
# Retrieval + Summarization function
# ------------------------------------------------------------------------------
def retrieve_context_and_summary(query: str, top_k: int = 5):
    # 1) Embed the user query
    embed_resp = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    query_vector = embed_resp['embedding']

    # 2) Query Pinecone for nearest neighbors
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    contexts = [match["metadata"]["text"] for match in result["matches"]]

    # 3) Assemble context and prompt
    context_block = "\n\n---\n\n".join(contexts)
    prompt = f"""
You are an intelligent assistant. Use the following context to answer the user's question:

User question:
{query}

Context:
{context_block}

Answer:
"""
    # 4) Call Gemini model - UPDATED CODE HERE
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    
    # Extract text from response
    answer_text = response.text
    
    return contexts, answer_text
# Rest of your code remains the same...
# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Summarizer", layout="wide")
st.title("📄 PDF RAG Summarizer")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of context chunks:", min_value=1, max_value=10, value=5)
show_context = st.sidebar.checkbox("Show retrieved context", value=False)

query = st.text_area("Enter your question or summary request here:", height=150)

if st.button("Submit"):
    if not query.strip():
        st.error("Please enter a question or prompt.")
    else:
        with st.spinner("Retrieving context and generating summary..."):
            try:
                contexts, summary = retrieve_context_and_summary(query, top_k)
                st.subheader("Results")
                st.write(summary)

                if show_context:
                    st.subheader("📚 Retrieved Contexts")
                    for i, ctx in enumerate(contexts, 1):
                        st.markdown(f"**Context {i}:**")
                        st.write(ctx)

            except Exception as e:
                st.error(f"Error: {e}")



# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------    
# st.markdown(
#     """
#     <style>
#     footer {visibility: hidden;}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# st.markdown(
#     """
#     <style>
#     footer {visibility: hidden;}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
