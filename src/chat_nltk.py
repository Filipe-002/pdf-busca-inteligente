import textwrap
import sys
import re

import ollama

from search import query_docs, make_snippet
from extract import extract_all_texts

SYSTEM_BEHAVIOR = (
    "Você responde APENAS com base no contexto fornecido.\n"
    "Se a resposta não estiver no contexto, diga: 'Não encontrei nos documentos.'\n"
    "Se citar algo, mencione o(s) documento(s) entre colchetes (ex.: [Lei_9394_20121996.pdf])."
)

def strip_reasoning(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL|re.IGNORECASE)
    return text.strip()

def _get_model_ctx_tokens(model_name: str, override_ctx: int | None = None) -> int:
    """
    Tenta descobrir o tamanho do contexto (em tokens) do modelo no Ollama.
    """
    if override_ctx:
        return int(override_ctx)
    try:
        info = ollama.show(model=model_name)
        for key in ("context_length", "num_ctx", "ctx", "context"):
            if key in info:
                return int(info[key])
        params = info.get("parameters") or info.get("model_info") or {}
        for key in ("context_length", "num_ctx", "ctx", "context"):
            if key in params:
                return int(params[key])
    except Exception:
        pass
    return 4096

def _tokens_to_chars(tokens: int, chars_per_token: float = 3.5) -> int:
    """Heurística PT-BR: ~3.5 chars/token."""
    return int(tokens * chars_per_token)

def _budget_for_context(model_name: str, top_docs: int, override_ctx: int | None = None) -> tuple[int, int]:
    """
    Retorna (hard_limit_total_chars, max_chars_per_doc) adaptados ao modelo escolhido.
    Reservamos ~70% do contexto para o 'Contexto' (o resto fica para instruções e resposta).
    """
    ctx_tokens = _get_model_ctx_tokens(model_name, override_ctx=override_ctx)
    total_chars = _tokens_to_chars(ctx_tokens)
    hard_limit_total = int(total_chars * 0.70) 
    max_chars_per_doc = min(1200, max(300, hard_limit_total // max(1, top_docs)))
    return hard_limit_total, max_chars_per_doc

def _slice_around_snippet(full_text: str, snippet: str, width: int) -> str:
    if not full_text:
        return ""
    if not snippet:
        return full_text[:width]
    key = snippet[: min(80, len(snippet))]  
    m = re.search(re.escape(key), full_text, flags=re.IGNORECASE)
    if not m:
        return full_text[:width]
    mid = (m.start() + m.end()) // 2
    half = width // 2
    start = max(0, mid - half)
    end = min(len(full_text), start + width)
    return full_text[start:end]

def _build_context(
    docs_texts: dict,
    hits,
    model_name: str,
    max_docs: int,
    override_ctx_tokens: int | None = None
) -> str:
    hard_limit_total, max_chars_per_doc = _budget_for_context(
        model_name, top_docs=max_docs, override_ctx=override_ctx_tokens
    )

    ctx_parts, total = [], 0
    for fname, _score, snip in hits[:max_docs]:
        raw = docs_texts.get(fname, "")
        if not raw:
            continue
        chunk = _slice_around_snippet(raw, snip, width=max_chars_per_doc)
        part = f"[DOC: {fname}]\n{chunk}".strip()
        if total + len(part) > hard_limit_total:
            break
        ctx_parts.append(part)
        total += len(part)
    return "\n\n".join(ctx_parts)

def ask(question: str, top: int = 5, model: str = "deepseek-r1:1.5b", ctx_override: int | None = None) -> str:
    resp = ollama.chat(
    model=model,
    messages=[
        {"role": "system", "content": "Você é técnico, conciso, NÃO mostre seu raciocínio (não use <think>). Responda só o resultado final e cite fontes quando fizer sentido."},
        {"role": "user", "content": prompt}
    ],
)
    answer = resp["message"]["content"]
    return strip_reasoning(answer)


    hits = query_docs(question, top_n=top)
    if not hits:
        return "Não encontrei conteúdo relevante nos documentos."

    docs_texts = extract_all_texts(save_txt=False)
    context = _build_context(
        docs_texts, hits, model_name=model, max_docs=top, override_ctx_tokens=ctx_override
    )
    if not context.strip():
        return "Não encontrei conteúdo relevante nos documentos."

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
        return resp["message"]["content"]
    except Exception as e:
        return f"[ERRO ao chamar o modelo '{model}']: {e}"

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Chat RAG local (NLTK/BM25 + Ollama) com contexto adaptativo")
    p.add_argument("--q", required=True, help="Pergunta do usuário")
    p.add_argument("--top", type=int, default=5, help="Quantos documentos usar no contexto")
    p.add_argument("--model", default="deepseek-r1:1.5b", help="Nome do modelo no Ollama (ex.: llama3:8b)")
    p.add_argument("--ctx", type=int, help="(Opcional) sobrescrever o tamanho de contexto do modelo (tokens)")
    args = p.parse_args()

    answer = ask(args.q, top=args.top, model=args.model, ctx_override=args.ctx)
    sys.stdout.write(answer + "\n")
