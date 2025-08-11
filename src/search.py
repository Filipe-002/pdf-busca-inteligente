## essa classe é responsável pela busca por linguagem natural através do prompt

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import math
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

from extract import extract_all_texts  


INDEX_DIR = Path(__file__).parent.parent / "nltk_index"
INDEX_DIR.mkdir(exist_ok=True)
INDEX_PATH = INDEX_DIR / "bm25_index.pkl"

K1 = 1.5
B  = 0.75


STOPWORDS_PT = set(stopwords.words("portuguese"))
STEMMER = RSLPStemmer()

def tokenize_pt(text: str) -> List[str]:
    tokens = word_tokenize(text, language="portuguese")
    tokens = [t.lower() for t in tokens if re.search(r"\w", t)]
    tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]
    return tokens

class BM25Index:
    def __init__(self):
        self.doc_ids: List[str] = []              
        self.doc_len: Dict[int, int] = {}         
        self.avgdl: float = 0.0                   
        self.df: Dict[str, int] = {}              
        self.tf: Dict[int, Dict[str, int]] = {}   
        self.N: int = 0                           
        self.inverted: Dict[str, List[Tuple[int, int]]] = {}

    @staticmethod
    def _count_terms(tokens: List[str]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for t in tokens:
            d[t] = d.get(t, 0) + 1
        return d

    def build(self, docs: Dict[str, str]) -> None:
        self.doc_ids = list(docs.keys())
        self.N = len(self.doc_ids)
        total_len = 0

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
            for term, f in tf_doc.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.inverted.setdefault(term, []).append((doc_idx, f))

        self.avgdl = (total_len / self.N) if self.N else 0.0

    def _idf(self, term: str) -> float:
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

        scores: Dict[int, float] = {}

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

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.doc_ids[doc_id], float(score)) for doc_id, score in ranked]

def save_index(index: BM25Index):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"Índice BM25 salvo em: {INDEX_PATH}")

def load_index() -> BM25Index:
    try:
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Falha ao carregar índice BM25 ({e}). Recriando a partir dos PDFs...")
        from extract import extract_all_texts
        docs = extract_all_texts(save_txt=False)
        idx = BM25Index()
        idx.build(docs)
        save_index(idx)
        return idx

def make_snippet(text: str, query: str, width: int = 240) -> str:
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


def build_from_data() -> None:
    print("== Extraindo textos ==")
    docs = extract_all_texts() 
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
