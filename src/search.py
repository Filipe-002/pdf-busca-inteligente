# src/search_nltk.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import math
import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

from extract import extract_all_texts  # usa sua extração

# --- Configs ---
INDEX_DIR = Path(__file__).parent.parent / "nltk_index"
INDEX_DIR.mkdir(exist_ok=True)
INDEX_PATH = INDEX_DIR / "bm25_index.pkl"

# BM25 hiperparâmetros
K1 = 1.5
B  = 0.75

# Stopwords e stemmer PT-BR
STOPWORDS_PT = set(stopwords.words("portuguese"))
STEMMER = RSLPStemmer()

# Tokenização com NLTK (mantendo acentos)
def tokenize_pt(text: str) -> List[str]:
    # word_tokenize usa 'punkt' – já baixamos acima
    tokens = word_tokenize(text, language="portuguese")
    # Normaliza: minúsculas, remove tokens não alfanuméricos
    tokens = [t.lower() for t in tokens if re.search(r"\w", t)]
    # Remove stopwords e tokens muito curtos
    tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]
    # Stemming PT-BR
    tokens = [STEMMER.stem(t) for t in tokens]
    return tokens

# ---------- Índice BM25 ----------
class BM25Index:
    def __init__(self):
        # dicionários
        self.doc_ids: List[str] = []              # ordem dos docs
        self.doc_len: Dict[int, int] = {}         # doc_id -> tamanho (em termos)
        self.avgdl: float = 0.0                   # comprimento médio
        self.df: Dict[str, int] = {}              # termo -> docs contendo o termo
        self.tf: Dict[int, Dict[str, int]] = {}   # doc_id -> { termo: freq }
        self.N: int = 0                           # nº de docs
        # índice invertido: termo -> lista de (doc_id, freq)
        self.inverted: Dict[str, List[Tuple[int, int]]] = {}

    @staticmethod
    def _count_terms(tokens: List[str]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for t in tokens:
            d[t] = d.get(t, 0) + 1
        return d

    def build(self, docs: Dict[str, str]) -> None:
        """
        docs: { "arquivo.pdf": "texto completo" }
        """
        self.doc_ids = list(docs.keys())
        self.N = len(self.doc_ids)
        total_len = 0

        # zera estruturas
        self.doc_len.clear()
        self.df.clear()
        self.tf.clear()
        self.inverted.clear()

        for doc_idx, fname in enumerate(self.doc_ids):
            text = docs[fname]
            tokens = tokenize_pt(text)
            tf_doc = self._count_terms(tokens)
            self.tf[doc_idx] = tf_doc
            dl = sum(tf_doc.values())
            self.doc_len[doc_idx] = dl
            total_len += dl

            # atualiza df e invertido
            for term, f in tf_doc.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.inverted.setdefault(term, []).append((doc_idx, f))

        self.avgdl = (total_len / self.N) if self.N else 0.0

    def _idf(self, term: str) -> float:
        # BM25 idf com ajuste +1
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[Tuple[str, float]]:
        """
        Retorna [(nome_pdf, score), ...] ordenado por relevância (desc).
        """
        q_terms = tokenize_pt(query)
        if not q_terms:
            return []

        # acumula score por doc
        scores: Dict[int, float] = {}

        # ótimiz: itere sobre postings de cada termo da query
        for q in q_terms:
            postings = self.inverted.get(q)
            if not postings:
                continue
            idf = self._idf(q)
            for doc_id, f_td in postings:
                dl = self.doc_len.get(doc_id, 0)
                denom = f_td + K1 * (1 - B + B * (dl / (self.avgdl or 1.0)))
                s = idf * ((f_td * (K1 + 1)) / (denom or 1e-9))
                scores[doc_id] = scores.get(doc_id, 0.0) + s

        # ordena por score desc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[doc_id], float(score)) for doc_id, score in ranked]

# ---------- Persistência ----------
def save_index(index: BM25Index):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"Índice BM25 salvo em: {INDEX_PATH}")

def load_index() -> BM25Index:
    if not INDEX_PATH.exists():
        raise FileNotFoundError("Índice não encontrado. Rode: python src/search_nltk.py --build")
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)

# ---------- Snippet ----------
def make_snippet(text: str, query: str, width: int = 240) -> str:
    """Gera um trechinho do doc contendo a primeira palavra da query (pré-processamento simples)."""
    # pega primeira palavra "significativa" da query (sem stopwords)
    q_tokens = [t for t in word_tokenize(query, language="portuguese") if t.lower() not in STOPWORDS_PT]
    if not q_tokens:
        return text[:width] + "..."
    key = re.escape(q_tokens[0])
    m = re.search(key, text, flags=re.IGNORECASE)
    if not m:
        return text[:width] + "..."
    i = max(m.start() - width // 2, 0)
    j = min(i + width, len(text))
    return ("..." if i > 0 else "") + text[i:j] + ("..." if j < len(text) else "")

# ---------- Build / Search (CLI) ----------
def build_from_data() -> None:
    print("== Extraindo textos ==")
    docs = extract_all_texts()  # reusa sua extração
    print(f"Total de documentos: {len(docs)}")

    print("== Construindo índice BM25 (NLTK) ==")
    idx = BM25Index()
    idx.build(docs)
    save_index(idx)

def query_docs(q: str, top_n: int = 5) -> List[Tuple[str, float, str]]:
    idx = load_index()
    ranking = idx.score(q)
    if not ranking:
        return []

    # Carrega textos para gerar snipets (rápido com poucos docs)
    docs = extract_all_texts(save_txt=False)
    results = []
    for fname, score in ranking[:top_n]:
        text = docs.get(fname, "")
        snippet = make_snippet(text, q)
        results.append((fname, score, snippet))
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Busca semântica clássica com NLTK + BM25")
    p.add_argument("--build", action="store_true", help="Reconstrói o índice a partir dos PDFs")
    p.add_argument("--query", type=str, help="Consulta em linguagem natural")
    p.add_argument("--top", type=int, default=5, help="Quantos documentos retornar")
    args = p.parse_args()

    if args.build:
        build_from_data()

    if args.query:
        hits = query_docs(args.query, top_n=args.top)
        if not hits:
            print("Nenhum resultado.")
        else:
            print("\n== Resultados ==")
            for fname, score, snip in hits:
                print(f"[{fname}] BM25={score:.4f}\n{snip}\n")
