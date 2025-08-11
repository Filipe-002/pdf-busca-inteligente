import re
import textwrap
import streamlit as st

from search import build_from_data, query_docs, make_snippet
from extract import extract_all_texts

try:
    import ollama
except Exception:
    ollama = None

st.set_page_config(page_title="PDFScanner: Pergunte à IA!", page_icon="📄", layout="wide")
st.title("📄 PDFScanner: Pergunte à IA!")

MODEL_NAME = "deepseek-r1:1.5b"
TOP_N = 9999
CTX_CHARS = 800
HARD_LIMIT_TOTAL = 4000

def strip_reasoning(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'(?is)<think>.*?</think>\s*', '', text)
    text = re.sub(r'(?is)<think>.*$', '', text)
    text = re.sub(r'(?is)^.*?</think>\s*', '', text)
    text = re.sub(r'(?im)^\s*(thoughts?|reasoning|scratchpad)\s*:.*?(?:\n\s*\n|$)', '\n', text)
    return text.strip()

def highlight(text: str, terms):
    out = text
    for t in terms:
        if not t.strip():
            continue
        out = re.sub(fr"({re.escape(t)})", r"**\1**", out, flags=re.IGNORECASE)
    return out

def build_context_from_hits(question: str, hits, per_doc_chars: int, total_limit: int):
    docs_texts = extract_all_texts(save_txt=False)
    parts = []
    used = 0
    total = 0
    for fname, _score, _snip in hits:
        raw = docs_texts.get(fname, "")
        if not raw:
            continue
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
                {"role": "system", "content": "Você é técnico, conciso. NÃO mostre raciocínio (não use <think>). Responda apenas o resultado final e cite fontes quando fizer sentido."},
                {"role": "user", "content": prompt}
            ],
        )
        return strip_reasoning(resp["message"]["content"]), context
    except Exception as e:
        return f"[ERRO ao chamar o modelo '{model}']: {e}", ""

st.subheader("💬 Chat local (RAG com BM25 → Ollama)")
q_chat = st.text_area("Pergunte ao chat:", height=100, placeholder="Ex.: Quais são as principais diretrizes da LDB 9.394?")
run_chat = st.button("Perguntar", type="primary", key="btn_chat")
if run_chat:
    if not q_chat.strip():
        st.warning("Digite uma pergunta primeiro.")
    else:
        with st.spinner("Buscando trechos e gerando resposta..."):
            try:
                hits = query_docs(q_chat, top_n=TOP_N)
            except Exception as e:
                st.error(f"Erro ao consultar índice BM25: {e}")
                hits = []
            if not hits:
                st.warning("Nenhum contexto encontrado para a pergunta.")
            else:
                answer, used_context = ask_llm(
                    q_chat, hits, model=MODEL_NAME,
                    per_doc_chars=CTX_CHARS, total_limit=HARD_LIMIT_TOTAL
                )
                st.markdown("**Resposta:**")
                st.write(answer)
                with st.expander("🔎 Contexto usado (fontes)"):
                    st.code(used_context)
