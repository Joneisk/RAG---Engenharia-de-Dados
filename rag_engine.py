"""
Módulo RAG da Xperiun Tecnologia S.A.
Implementa:
- Ingestão de PDFs e arquivos de texto
- Chunking com tamanho 350 e overlap 40
- Embeddings OpenAI text-embedding-3-small
- Vector store Chroma efêmero (in-memory)
- Busca semântica (Top K=3)
- Geração com Claude Sonnet
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import chromadb

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv(override=True)


def load_xperiun_documents(
    docs_dir: str = "docs",
    context_dir: str = "contexto"
) -> List[Document]:
    """
    Carrega todos os documentos corporativos da pasta docs/ (.pdf)
    e opcionalmente da pasta contexto/ (.txt).
    """
    documents: List[Document] = []
    base_path = Path(".")

    # 1. Carregar PDFs da pasta docs
    docs_path = base_path / docs_dir
    if docs_path.exists() and docs_path.is_dir():
        for file_path in docs_path.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(file_path))
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    page_num = doc.metadata.get("page", 0)
                    doc.metadata["page"] = page_num + 1 if isinstance(page_num, int) else 1
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["doc_type"] = "PDF Corporativo"
                    documents.append(doc)
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")

    # 2. Carregar arquivo de contexto institucional se presente
    ctx_path = base_path / context_dir
    if ctx_path.exists() and ctx_path.is_dir():
        for file_path in ctx_path.glob("*.txt"):
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                txt_docs = loader.load()
                for doc in txt_docs:
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["doc_type"] = "Contexto Mestre Institucional"
                    doc.metadata["page"] = 1
                    documents.append(doc)
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")

    return documents


def create_chunks(
    documents: List[Document],
    chunk_size: int = 350,
    chunk_overlap: int = 40
) -> List[Document]:
    """
    Divide os documentos em chunks de acordo com as especificações da Onda 2:
    chunk_size = 350 caracteres, chunk_overlap = 40 caracteres.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Adicionar metadados enriquecidos para exibição e rastreamento
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1
        chunk.metadata["char_count"] = len(chunk.page_content)
    
    return chunks


def init_in_memory_vectorstore(
    chunks: List[Document],
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None
) -> Chroma:
    """
    Inicializa o Chroma DB em memória (efêmero) e indexa os chunks.
    Suporta OpenAI (text-embedding-3-small) e Google Gemini (models/text-embedding-004).
    """
    if embedding_provider == "gemini":
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Chave GEMINI_API_KEY / GOOGLE_API_KEY não informada ou ausente no .env.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model or "models/gemini-embedding-001",
            google_api_key=api_key
        )
    else:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Chave OPENAI_API_KEY não informada ou ausente no .env.")
        embeddings = OpenAIEmbeddings(
            model=embedding_model or "text-embedding-3-small",
            api_key=api_key
        )

    # Cliente Chroma efêmero (em memória)
    ephemeral_client = chromadb.EphemeralClient()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=ephemeral_client,
        collection_name="xperiun_knowledge_base"
    )
    
    return vectorstore


def search_semantic_chunks(
    vectorstore: Chroma,
    query: str,
    k: int = 3
) -> List[Tuple[Document, float]]:
    """
    Executa a busca semântica (Augmented) retornando os K chunks mais similares
    juntamente com seus respectivos scores de distância.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def format_context_for_prompt(chunks: List[Document]) -> str:
    """
    Formata os chunks recuperados em uma string clara com identificação das fontes.
    """
    formatted_pieces = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source_file", "Desconhecido")
        page = chunk.metadata.get("page", 1)
        chunk_id = chunk.metadata.get("chunk_id", idx)
        formatted_pieces.append(
            f"--- CHUNK #{chunk_id} [Origem: {source} | Página: {page}] ---\n"
            f"{chunk.page_content}\n"
        )
    return "\n".join(formatted_pieces)


def generate_llm_response(
    query: str,
    chunks: List[Document],
    provider: str = "gemini",
    anthropic_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Executa a etapa de Generation utilizando o provedor selecionado:
    - Google Gemini (gratuito via Google AI Studio)
    - Anthropic Claude (Sonnet)
    """
    context_str = format_context_for_prompt(chunks)
    system_message = SYSTEM_PROMPT.format(context=context_str)
    user_message = USER_PROMPT_TEMPLATE.format(question=query)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    if provider.lower() == "gemini":
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Chave GEMINI_API_KEY / GOOGLE_API_KEY não informada ou ausente no .env.")

        llm = ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )
    else:
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Chave ANTHROPIC_API_KEY não informada ou ausente no .env.")

        llm = ChatAnthropic(
            model=model_name or "claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key,
            temperature=0.0,
            max_tokens=1500
        )

    response = llm.invoke(messages)
    return response.content
