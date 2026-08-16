"""Streamlit PDF RAG demo powered entirely by local Ollama models."""

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

IS_CLOUD = bool(OLLAMA_API_KEY)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "https://ollama.com" if IS_CLOUD else "http://localhost:11434",
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gpt-oss:20b" if IS_CLOUD else "llama3.2",
)

EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text",
)

CLIENT_KWARGS = (
    {
        "headers": {
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        }
    }
    if IS_CLOUD
    else {}
)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Generate five different versions of the user's question for document retrieval.
The variations should preserve the original meaning while using different wording or perspectives.
Return one question per line and nothing else.

Original question: {question}""",
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not supported by the context, say that you cannot find it in the uploaded document.
Be concise, but include enough detail to be useful.

Context:
{context}

Question: {question}
"""
)


def format_docs(documents) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_bytes: bytes, file_hash: str):
    """Load, chunk, and embed a PDF. file_hash is used as a cache key."""
    #del file_hash

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = Path(temp_file.name)

        documents = PyMuPDFLoader(str(temp_path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
        )
        chunks = splitter.split_documents(documents)

        if IS_CLOUD:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            embeddings = OllamaEmbeddings(
                model=EMBED_MODEL,
                base_url=OLLAMA_HOST,
            )

        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"pdf-rag-{file_hash[:12]}",
        )
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def build_chain(vector_store):
    llm = ChatOllama(model=MODEL, base_url=OLLAMA_HOST, temperature=0.2)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        llm=llm,
        prompt=QUERY_PROMPT,
    )

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )


def main() -> None:
    st.set_page_config(page_title="Local PDF RAG", page_icon="📄")
    st.title("Local PDF RAG with Ollama")
    st.caption("Upload a PDF and ask questions about it. Processing stays on your machine.")

    with st.sidebar:
        st.subheader("Configuration")
        st.code(
            f"OLLAMA_HOST={OLLAMA_HOST}\n"
            f"OLLAMA_MODEL={MODEL}\n"
            f"OLLAMA_EMBED_MODEL={EMBED_MODEL}"
        )
        st.markdown(
            "Before starting, make sure Ollama is running and both models are installed."
        )

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is None:
        st.info("Upload a PDF to begin.")
        return

    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    try:
        with st.spinner("Indexing PDF locally..."):
            vector_store = build_vector_store(pdf_bytes, file_hash)
            chain = build_chain(vector_store)
    except Exception as exc:
        st.error(
            "Could not index the PDF. Check that Ollama is running and that "
            f"'{EMBED_MODEL}' is installed.\n\n{exc}"
        )
        return

    st.success(f"Ready: {uploaded_file.name}")

    question = st.chat_input("Ask a question about the PDF")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
            st.write(answer)
        except Exception as exc:
            st.error(f"Ollama request failed: {exc}")


if __name__ == "__main__":
    main()
