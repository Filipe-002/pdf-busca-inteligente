# src/app_streamlit.py
import re
import textwrap
import streamlit as st

# ==== integra seus módulos ====
from search import build_from_data, query_docs, make_snippet
from extract import extract_all_texts

# Ollama client (LLM local)
try:
    import ollama
except Exception:
    ollama = None

st.set_page_config(page_title="PDFs: Busca + Chat (local)", page_icon="📄", layout="wide")
st.title("📄 PDFs: Busca (NLTK+BM25) + Chat RAG (Ollama)")

st.caption(
    "• Busca clássica em PT-BR com NLTK (stopwords/stemming) e BM25  • "
    "Chat local via Ollama (modelo leve) respondendo **apenas com base** nos PDFs"
)

# ======== Sidebar ========
st.sidebar.header("Configurações")
top_n = st.sidebar.slider("Top-N documentos", 1, 10, 5)
ctx_chars = st.sidebar.slider("Caracteres por documento no contexto (chat)", 300, 2000, 900, 100)
model_name = st.sidebar.text_input("Modelo do Ollama", value="deepseek-r1:1.5b", help="Use o nome que aparece em `ollama list`")
hard_limit_total = st.sidebar.slider("Limite total de contexto", 1000, 8000, 4000, 250)

if st.sidebar.button("🔧 Reconstruir índice agora"):
    with st.spinner("Reconstruindo índice BM25 a partir dos PDFs..."):
        build_from_data()
    st.success("Índice reconstruído com sucesso!")

# ======== Helpers (highlight + contexto) ========
def highlight(text: str, terms):
    out = text
    for t in terms:
        if not t.strip():
            continue
        out = re.sub(fr"({re.escape(t)})", r"**\1**", out, flags=re.IGNORECASE)
    return out

def build_context_from_hits(question: str, hits, per_doc_chars: int, total_limit: int):
    """Monta um contexto compacto a partir dos documentos top-N (com limite total)."""
    docs_texts = extract_all_texts(save_txt=False)
    parts = []
    used = 0
    total = 0
    for fname, _score, _snip in hits:
        raw = docs_texts.get(fname, "")
        if not raw:
            continue
        # Usa snippet focado na pergunta para melhorar relevância do contexto
        snip = make_snippet(raw, question, width=per_doc_chars)
        chunk = f"[DOC: {fname}]\n{snip}".strip()
        if total + len(chunk) > total_limit:
            break
        parts.append(chunk)
        used += 1
        total += len(chunk)
    return "\n\n".join(parts), used

SYSTEM_BEHAVIOR = (
    "Você responde APENAS com base no contexto fornecido.\n"
    "Se a resposta não estiver no contexto, diga: 'Não encontrei nos documentos.'\n"
    "Quando fizer sentido, cite o(s) documento(s) entre colchetes, ex.: [Lei_9394_20121996.pdf]."
)

def ask_llm(question: str, hits, model: str, per_doc_chars: int, total_limit: int):
    if ollama is None:
        return "O pacote 'ollama' não está instalado neste ambiente.", ""
    context, used = build_context_from_hits(question, hits, per_doc_chars, total_limit)
    if not context.strip():
        return "Não encontrei conteúdo relevante nos documentos.", ""
    prompt = textwrap.dedent(f"""
    {SYSTEM_BEHAVIOR}

    # Pergunta
    {question}

    # Contexto
    {context}
    """).strip()
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Você é técnico, conciso e cita fontes quando fizer sentido."},
                {"role": "user", "content": prompt}
            ],
        )
        return resp["message"]["content"], context
    except Exception as e:
        return f"[ERRO ao chamar o modelo '{model}']: {e}", ""

# ======== Layout com abas ========
tab_busca, tab_chat = st.tabs(["🔎 Busca", "💬 Chat"])

with tab_busca:
    st.subheader("Busca por linguagem natural (NLTK + BM25)")
    q = st.text_input("O que você procura? (ex.: diretrizes da educação básica, competências do CNE, etc.)", key="q_busca")
    col1, col2 = st.columns([1, 3])
    run_search = col1.button("Buscar", type="primary", key="btn_busca")

    if run_search:
        with st.spinner("Consultando índice..."):
            hits = query_docs(q, top_n=top_n)
        if not hits:
            st.warning("Nenhum resultado encontrado. Tente reformular a consulta.")
        else:
            # termos para highlight (simples: separa por espaços)
            terms = [t for t in re.split(r"\s+", q.strip()) if len(t) > 1][:6]
            st.write(f"**{len(hits)}** documento(s) mais relevantes:\n")
            for fname, score, snip in hits:
                with st.expander(f"📄 {fname}  —  BM25={score:.4f}", expanded=False):
                    st.markdown(highlight(snip, terms))

with tab_chat:
    st.subheader("Chat local (RAG com BM25 → Ollama)")
    q_chat = st.text_area("Pergunta para o chat:", height=100, placeholder="Ex.: Quais são as principais diretrizes da LDB 9.394?")
    run_chat = st.button("Perguntar ao LLM", type="primary", key="btn_chat")

    if run_chat:
        with st.spinner("Buscando trechos e gerando resposta..."):
            hits = query_docs(q_chat, top_n=top_n)
            if not hits:
                st.warning("Nenhum contexto encontrado para a pergunta.")
            else:
                answer, used_context = ask_llm(
                    q_chat, hits, model=model_name,
                    per_doc_chars=ctx_chars, total_limit=hard_limit_total
                )
                st.markdown("**Resposta:**")
                st.write(answer)
                with st.expander("🔍 Contexto usado (fontes)"):
                    st.code(used_context)
                st.caption("Dica: ajuste 'Top-N documentos', 'Caracteres por documento' e 'Limite total' na barra lateral se a resposta vier muito vaga ou truncada.")
