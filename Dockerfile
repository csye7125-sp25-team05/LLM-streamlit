FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY LLM.py .
COPY .env .

# Create a Streamlit config file with proper string formatting
RUN mkdir -p /root/.streamlit/
RUN echo '\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
baseUrlPath = "/rag"\n\
' > /root/.streamlit/config.toml

ENV PINECONE_API_KEY=""
ENV PINECONE_ENVIRONMENT="us-east1-gcp"
ENV PINECONE_INDEX_NAME="pdf-embeddings"
ENV GENIE_API_KEY=""

EXPOSE 8501

CMD ["streamlit", "run", "LLM.py", "--server.address=0.0.0.0"]
