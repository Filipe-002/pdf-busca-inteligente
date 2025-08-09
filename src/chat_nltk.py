from __future__ import annotations
from typing import List, Tuple
import textwrap
import sys

import ollama

# Reaproveita sua busca clássica (NLTK + BM25)
# Atenção: ajuste o import conforme o nome do seu arquivo (search.py ou search_nltk.py)
from search import query_docs
from extract import extract_all_texts

SYSTEM_BEHAVIOR = (
    "Você responde APENAS com base no contexto fornecido.\n"
    "Se a resposta não estiver no contexto, diga: 'Não encontrei nos documentos.'\n"
    "Se citar algo, mencione o(s) documento(s) entre colchetes (ex.: [Lei_9394_20121996.pdf])."
)

def _build_context(docs_texts: dict, hits: List[Tuple[str, float, str]],
                   max_docs: int = 5, max_chars_per_doc: int = 900,
                   hard_limit_total: int = 4000) -> str:
    """
    Junta trechos de vários documentos em um único contexto, com limites para caber em modelos pequenos.
    """
    ctx_parts = []
    total = 0
    used = 0
    for fname, _score, _snip in hits[:max_docs]:
        # pega o texto bruto do doc (já extraído por você)
        text = docs_texts.get(fname, "")
        if not text:
            continue
        # corta um pedaço inicial (ou você pode escolher um trecho mais representativo)
        chunk = text[:max_chars_per_doc]
        part = f"[DOC: {fname}]\n{chunk}".strip()
        if total + len(part) > hard_limit_total:
            break
        ctx_parts.append(part)
        total += len(part)
        used += 1
    if not ctx_parts:
        return ""
    return "\n\n".join(ctx_parts)

def ask(question: str, top: int = 5, model: str = "deepseek:1b") -> str:
    """
    Faz busca BM25 -> monta contexto -> chama Ollama (modelo local).
    """
    # 1) Recupera docs mais relevantes
    hits = query_docs(question, top_n=top)
    if not hits:
        return "Não encontrei conteúdo relevante nos documentos."

    # 2) Carrega textos de todos os docs (uma vez) e monta contexto compacto
    docs_texts = extract_all_texts(save_txt=False)
    context = _build_context(docs_texts, hits, max_docs=top)

    if not context.strip():
        return "Não encontrei conteúdo relevante nos documentos."

    # 3) Monta o prompt final
    prompt = textwrap.dedent(f"""
    {SYSTEM_BEHAVIOR}

    # Pergunta
    {question}

    # Contexto
    {context}
    """).strip()

    # 4) Chama o modelo local via Ollama
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Você é técnico, conciso e cita fontes quando fizer sentido."},
                {"role": "user", "content": prompt}
            ],
        )
        return resp["message"]["content"]
    except Exception as e:
        return f"[ERRO ao chamar o modelo '{model}']: {e}"

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Chat RAG local (NLTK/BM25 + Ollama)")
    p.add_argument("--q", required=True, help="Pergunta do usuário")
    p.add_argument("--top", type=int, default=5, help="Quantos documentos usar no contexto")
    p.add_argument("--model", default="deepseek:1b", help="Nome do modelo no Ollama (ex.: deepseek:1b)")
    args = p.parse_args()

    answer = ask(args.q, top=args.top, model=args.model)
    sys.stdout.write(answer + "\n")
