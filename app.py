"""
Plataforma RAG - Xperiun Tecnologia S.A.
Visualizador e Chat Corporativo em 5 Ondas (LangChain + Chroma In-Memory + Claude Sonnet + Streamlit)
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env
load_dotenv(override=True)

# Importar motor RAG e prompts
from rag_engine import (
    load_xperiun_documents,
    create_chunks,
    init_in_memory_vectorstore,
    search_semantic_chunks,
    generate_llm_response,
    format_context_for_prompt
)
from prompts import SYSTEM_PROMPT

# Configuração da página Streamlit
st.set_page_config(
    page_title="Xperiun | Chat Corporativo RAG",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .chunk-box {
        background-color: #F1F5F9;
        border-left: 4px solid #2563EB;
        padding: 12px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    .source-tag {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Inicialização de Session State
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# --- BARRA LATERAL (CONFIGURAÇÕES & GESTÃO DO RAG) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cloud-checked.png", width=64)
    st.title("Xperiun Knowledge Hub")
    st.caption("Arquitetura RAG Corporativa em 5 Ondas")
    
    st.divider()
    st.subheader("⚙️ Provedores & Configurações de API")
    
    # Seleção de Provedores
    llm_choice = st.selectbox(
        "🤖 Provedor de IA (Generation):",
        ["Google Gemini (Gratuito)", "Anthropic Claude (Sonnet 3.5)"],
        index=0,
        help="O Google Gemini oferece cota gratuita via Google AI Studio."
    )
    
    embedding_choice = st.selectbox(
        "🧠 Provedor de Embeddings:",
        ["OpenAI (text-embedding-3-small)", "Google Gemini (gemini-embedding-001 - Gratuito)"],
        index=0
    )

    env_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    env_openai_key = os.getenv("OPENAI_API_KEY", "")
    env_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Campos de Chaves
    gemini_key_input = st.text_input(
        "Google Gemini API Key:",
        value=env_gemini_key,
        type="password",
        help="Obtenha gratuitamente em: https://aistudio.google.com/app/apikey"
    )
    
    openai_key_input = st.text_input(
        "OpenAI API Key (Embeddings):",
        value=env_openai_key,
        type="password",
        help="Utilizada para gerar os embeddings caso escolha OpenAI"
    )
    
    anthropic_key_input = st.text_input(
        "Anthropic API Key (Claude):",
        value=env_anthropic_key,
        type="password",
        help="Utilizada para o modelo Claude Sonnet"
    )
    
    if "Gemini" in llm_choice and not gemini_key_input:
        st.info("💡 Pegue sua chave grátis do Gemini em [Google AI Studio](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    st.subheader("📏 Parâmetros de Chunking (Onda 2)")
    chunk_size = st.number_input("Tamanho do Chunk (chars):", min_value=100, max_value=1000, value=350, step=50)
    chunk_overlap = st.number_input("Overlap (chars):", min_value=0, max_value=200, value=40, step=10)
    top_k = st.slider("Top K Chunks Recuperados (Ondas 4 & 5):", min_value=1, max_value=10, value=3)

    st.divider()
    
    # Botão de Carregamento e Processamento
    btn_process = st.button("🚀 Processar Documentos & Criar Chunks", use_container_width=True, type="primary")

    if btn_process:
        with st.spinner("Lendo arquivos da pasta docs/ e contexto/..."):
            docs = load_xperiun_documents()
            st.session_state.raw_docs = docs
            
            if docs:
                chunks = create_chunks(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.session_state.chunks = chunks
                st.success(f"Sucesso! {len(docs)} documentos e {len(chunks)} chunks gerados.")
                
                # Inicializar o Chroma com o provedor escolhido
                is_gemini_embed = "Gemini" in embedding_choice
                active_embed_key = gemini_key_input if is_gemini_embed else openai_key_input
                
                if active_embed_key:
                    with st.spinner(f"Indexando chunks no Chroma DB via {embedding_choice}..."):
                        try:
                            vstore = init_in_memory_vectorstore(
                                chunks=chunks,
                                openai_api_key=openai_key_input,
                                gemini_api_key=gemini_key_input,
                                embedding_provider="gemini" if is_gemini_embed else "openai",
                                embedding_model="models/gemini-embedding-001" if is_gemini_embed else "text-embedding-3-small"
                            )
                            st.session_state.vectorstore = vstore
                            st.info("Chroma DB (Efêmero) inicializado e pronto para busca semântica!")
                        except Exception as err:
                            st.error(f"Erro ao inicializar Chroma DB: {err}")
                else:
                    st.warning(f"Insira sua API Key ({'Google Gemini' if is_gemini_embed else 'OpenAI'}) para habilitar a busca semântica (Chroma DB).")
            else:
                st.error("Nenhum documento encontrado na pasta docs/.")

    st.divider()
    st.caption("© 2026 Xperiun Tecnologia S.A. | RAG Platform")


# --- CABEÇALHO PRINCIPAL ---
st.markdown('<div class="main-header">Xperiun Tecnologia S.A. | RAG Knowledge Hub</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Plataforma interna para auditoria de chunks, busca semântica transparente e respostas embasadas via IA (Gemini / Claude).</div>',
    unsafe_allow_html=True
)

# Auto-carregamento inicial dos documentos se ainda não carregados
if not st.session_state.raw_docs:
    st.session_state.raw_docs = load_xperiun_documents()
    if st.session_state.raw_docs and not st.session_state.chunks:
        st.session_state.chunks = create_chunks(st.session_state.raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# --- ABAS DE EVOLUÇÃO DAS 5 ONDAS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 Onda 2 & 3: Visualização dos Chunks",
    "🔍 Onda 4: Augmented (Busca Semântica)",
    "💬 Onda 5: Generation (Chat Corporativo)",
    "ℹ️ Governança & System Prompt"
])


# ==============================================================================
# TAB 1: VISUALIZAÇÃO DOS CHUNKS (ONDAS 2 & 3)
# ==============================================================================
with tab1:
    st.subheader("Inspeção e Auditoria dos Chunks Gerados")
    st.markdown("Nesta etapa, visualizamos a fragmentação de 350 caracteres com overlap de 40 sobre os documentos corporativos da Xperiun.")
    
    chunks = st.session_state.chunks
    docs = st.session_state.raw_docs
    
    if chunks:
        # Métricas no topo
        c1, c2, c3, c4 = st.columns(4)
        total_chars = sum(c.metadata.get("char_count", len(c.page_content)) for c in chunks)
        avg_chars = round(total_chars / len(chunks), 1) if chunks else 0
        
        c1.metric("Documentos Carregados", len(docs))
        c2.metric("Total de Chunks Gerados", len(chunks))
        c3.metric("Média de Caracteres/Chunk", f"{avg_chars} chars")
        c4.metric("Overlap Configurado", f"{chunk_overlap} chars")

        st.markdown("---")

        # Filtro e Seleção de Chunks
        col_filter, col_inspect = st.columns([1, 2])
        
        with col_filter:
            st.markdown("#### 📁 Origem dos Chunks")
            sources = sorted(list(set(c.metadata.get("source_file", "Desconhecido") for c in chunks)))
            selected_source = st.selectbox("Filtrar por Documento de Origem:", ["Todos"] + sources)
            
            filtered_chunks = [
                c for c in chunks 
                if selected_source == "Todos" or c.metadata.get("source_file") == selected_source
            ]
            
            st.write(f"Mostrando **{len(filtered_chunks)}** chunks de **{selected_source}**")
            
            chunk_options = [
                f"Chunk #{c.metadata.get('chunk_id')} | {c.metadata.get('source_file')} (pág. {c.metadata.get('page', 1)})" 
                for c in filtered_chunks
            ]
            selected_chunk_str = st.selectbox("Selecione um chunk para inspecionar:", chunk_options)
            
            # Tabela Resumo dos Chunks
            df_chunks = pd.DataFrame([
                {
                    "ID": c.metadata.get("chunk_id"),
                    "Arquivo": c.metadata.get("source_file"),
                    "Pág.": c.metadata.get("page", 1),
                    "Caracteres": c.metadata.get("char_count", len(c.page_content)),
                    "Prévia": c.page_content[:60] + "..."
                }
                for c in filtered_chunks
            ])
            st.dataframe(df_chunks, use_container_width=True, height=280)

        with col_inspect:
            st.markdown("#### 🔬 Conteúdo do Chunk Selecionado")
            if selected_chunk_str:
                chunk_id_val = int(selected_chunk_str.split("|")[0].replace("Chunk #", "").strip())
                selected_chunk = next((c for c in chunks if c.metadata.get("chunk_id") == chunk_id_val), None)
                
                if selected_chunk:
                    st.markdown(
                        f"<span class='source-tag'>Documento: {selected_chunk.metadata.get('source_file')}</span> "
                        f"<span class='source-tag'>Página: {selected_chunk.metadata.get('page', 1)}</span> "
                        f"<span class='source-tag'>Tamanho: {len(selected_chunk.page_content)} caracteres</span>",
                        unsafe_allow_html=True
                    )
                    st.text_area(
                        "Texto Integral do Fragmento:",
                        value=selected_chunk.page_content,
                        height=220,
                        disabled=True
                    )
                    with st.expander("Metadados Técnicos"):
                        st.json(selected_chunk.metadata)
    else:
        st.info("Nenhum chunk disponível. Clique em 'Processar Documentos & Criar Chunks' na barra lateral.")


# ==============================================================================
# TAB 2: AUGMENTED / BUSCA SEMÂNTICA PURA (ONDA 4)
# ==============================================================================
with tab2:
    st.subheader("Onda 4: Busca Semântica Pura com Chroma DB (Sem IA)")
    st.markdown(
        "Nesta etapa, validamos o componente **Augmented** do RAG: a pergunta é convertida em embedding "
        "e recupera os **Top 3 chunks mais similares** diretamente do ChromaDB em memória, **sem utilizar nenhum LLM ainda**."
    )
    
    if st.session_state.vectorstore is None:
        st.warning("⚠️ O Chroma DB ainda não foi inicializado. Insira sua OpenAI API Key na barra lateral e clique em 'Processar Documentos & Criar Chunks'.")
    else:
        st.success("✅ Chroma DB (In-Memory) está ativo e com chunks indexados!")
        
        query_wave4 = st.text_input(
            "Digite uma pergunta para testar a similaridade semântica:",
            placeholder="Exemplo: Qual é o SLA de disponibilidade exigido dos fornecedores?",
            key="query_wave4"
        )
        
        btn_search = st.button("🔍 Executar Busca Semântica", type="primary")
        
        if btn_search and query_wave4:
            with st.spinner("Buscando chunks mais relevantes via embedding..."):
                results = search_semantic_chunks(st.session_state.vectorstore, query_wave4, k=top_k)
                
                st.markdown(f"### Top {len(results)} Chunks Retornados:")
                
                for rank, (doc, score) in enumerate(results, start=1):
                    with st.container():
                        st.markdown(f"#### 🏆 Posição #{rank} — Similaridade / Score: `{score:.4f}`")
                        st.markdown(
                            f"**Origem:** `{doc.metadata.get('source_file')}` | "
                            f"**Página:** `{doc.metadata.get('page', 1)}` | "
                            f"**Chunk ID:** `#{doc.metadata.get('chunk_id')}` | "
                            f"**Tamanho:** `{len(doc.page_content)} caracteres`"
                        )
                        st.markdown(f'<div class="chunk-box">{doc.page_content}</div>', unsafe_allow_html=True)
                        st.divider()


# ==============================================================================
# TAB 3: GENERATION / CHAT COMPLETO (ONDA 5)
# ==============================================================================
with tab3:
    is_gemini_llm = "Gemini" in llm_choice
    model_display_name = "Google Gemini 1.5 Flash (Gratuito)" if is_gemini_llm else "Anthropic Claude Sonnet 3.5"
    st.subheader(f"Onda 5: Geração de Resposta com {model_display_name}")
    st.markdown(
        f"Esta é a etapa completa do RAG utilizando **{model_display_name}**. "
        "Conforme o **PRD.md**, a busca **sempre exibe os K chunks recuperados** para verificação antes de exibir a resposta final."
    )
    
    # Histórico de Conversa
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "chunks" in msg and msg["chunks"]:
                with st.expander(f"🔎 Chunks Consultados da Busca Semântica (K={len(msg['chunks'])})", expanded=False):
                    for rank, c in enumerate(msg["chunks"], start=1):
                        st.markdown(f"**Chunk #{rank} - {c['source']} (pág. {c['page']})**")
                        st.caption(c["text"])
                        st.divider()
            st.markdown(msg["content"])
    
    # Entrada do Usuário
    user_prompt = st.chat_input("Pergunte algo sobre contratos, preços, SLAs ou fornecedores da Xperiun...")
    
    if user_prompt:
        # Exibir mensagem do usuário
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Validação do Vectorstore
        if st.session_state.vectorstore is None:
            with st.chat_message("assistant"):
                error_msg = "⚠️ O banco vetorial ainda não foi inicializado. Por favor, insira sua chave de API na barra lateral e clique em 'Processar Documentos'."
                st.warning(error_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        elif is_gemini_llm and not gemini_key_input:
            with st.chat_message("assistant"):
                error_msg = "⚠️ Google Gemini API Key não configurada. Por favor, informe sua chave no campo da barra lateral ou no arquivo .env (pegue grátis em https://aistudio.google.com/app/apikey)."
                st.warning(error_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        elif not is_gemini_llm and not anthropic_key_input:
            with st.chat_message("assistant"):
                error_msg = "⚠️ Anthropic API Key não configurada. Por favor, informe sua chave no campo da barra lateral ou no arquivo .env para chamar o Claude Sonnet."
                st.warning(error_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.chat_message("assistant"):
                with st.spinner("1/2: Recuperando K chunks mais relevantes..."):
                    results = search_semantic_chunks(st.session_state.vectorstore, user_prompt, k=top_k)
                    retrieved_docs = [doc for doc, score in results]
                
                # Exibição transparente dos K chunks (Requisito mandatório do PRD)
                with st.expander(f"🔎 Chunks Consultados da Busca Semântica (K={len(retrieved_docs)})", expanded=True):
                    chunks_payload = []
                    for rank, (doc, score) in enumerate(results, start=1):
                        source = doc.metadata.get("source_file", "Doc")
                        page = doc.metadata.get("page", 1)
                        st.markdown(f"**Trecho #{rank} | Fonte:** `{source}` (Página {page}) — *Score: {score:.4f}*")
                        st.info(doc.page_content)
                        chunks_payload.append({
                            "source": source,
                            "page": page,
                            "text": doc.page_content,
                            "score": float(score)
                        })
                
                with st.spinner(f"2/2: Gerando resposta embasada com {model_display_name}..."):
                    try:
                        provider_key = "gemini" if is_gemini_llm else "claude"
                        model_to_use = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") if is_gemini_llm else os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                        
                        answer = generate_llm_response(
                            query=user_prompt,
                            chunks=retrieved_docs,
                            provider=provider_key,
                            anthropic_api_key=anthropic_key_input,
                            gemini_api_key=gemini_key_input,
                            model_name=model_to_use
                        )
                        st.markdown(answer)
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "chunks": chunks_payload
                        })
                    except Exception as ex:
                        err_text = f"Erro ao consultar a IA ({provider_key}): {str(ex)}"
                        st.error(err_text)
                        st.session_state.chat_messages.append({"role": "assistant", "content": err_text})


# ==============================================================================
# TAB 4: GOVERNANÇA & SYSTEM PROMPT
# ==============================================================================
with tab4:
    st.subheader("Governança do Agente & System Prompt Oficial")
    st.markdown("Especificações e diretrizes de compliance documentadas em `Agents.md`.")
    
    st.markdown("#### System Prompt Aplicado ao Claude Sonnet:")
    st.code(SYSTEM_PROMPT, language="markdown")
    
    st.markdown("#### Contexto Institucional Carregado:")
    with st.expander("Visualizar Contexto Mestre da Xperiun"):
        ctx_file = Path("contexto/contexto.txt")
        if ctx_file.exists():
            st.text(ctx_file.read_text(encoding="utf-8"))
        else:
            st.write("Arquivo contexto/contexto.txt não encontrado.")
