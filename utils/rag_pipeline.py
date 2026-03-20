"""
RAG Pipeline — Using Groq (free, fast, Llama-3)
"""

from __future__ import annotations
from typing import Literal
import requests


def _split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.create_documents([text])


def _get_embeddings(provider: str, api_key: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key=api_key)
    else:
        # Fully local — no API key needed
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def _build_vector_store(docs, embeddings, store_type: str):
    if store_type == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS.from_documents(docs, embeddings)
    else:
        from langchain_community.vectorstores import Chroma
        return Chroma.from_documents(docs, embeddings)


def _call_groq(prompt: str, api_key: str) -> str:
    """Call Groq API directly — free tier, Llama-3 8B."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1500,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code == 401:
        raise ValueError("Invalid Groq API key. Get one free at https://console.groq.com/keys")
    if response.status_code != 200:
        raise ValueError(f"Groq API error {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]["content"]


class RAGPipeline:
    def __init__(
        self,
        resume_text: str,
        jd_text: str,
        api_key: str,
        provider: Literal["openai", "groq"] = "groq",
        model_name: str = "llama-3.1-8b-instant",
        vector_store_type: Literal["faiss", "chromadb"] = "faiss",
    ):
        self.resume_text = resume_text
        self.jd_text = jd_text
        self.provider = provider
        self.api_key = api_key

        self.embeddings = _get_embeddings(provider, api_key)

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.3, max_tokens=1500)
        else:
            self.llm = None  # Groq uses direct requests

        resume_docs = _split_text(f"[RESUME]\n{resume_text}")
        jd_docs = _split_text(f"[JOB DESCRIPTION]\n{jd_text}")
        self.vectorstore = _build_vector_store(resume_docs + jd_docs, self.embeddings, vector_store_type)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def analyze(self, prompt_template: str, task: str = "general") -> str:
        query_map = {
            "skill_gap": "required skills qualifications experience technologies tools",
            "resume_rewrite": "achievements accomplishments metrics impact results responsibilities",
            "cover_letter": "company culture mission role responsibilities requirements",
        }
        relevant_docs = self.retriever.invoke(query_map.get(task, "relevant experience skills"))
        context = "\n\n---\n\n".join(d.page_content for d in relevant_docs)
        filled_prompt = prompt_template.format(
            resume=self.resume_text[:3000],
            job_description=self.jd_text[:3000],
            retrieved_context=context,
        )
        if self.provider == "openai":
            response = self.llm.invoke(filled_prompt)
            return response.content if hasattr(response, "content") else str(response)
        else:
            return _call_groq(filled_prompt, self.api_key)
