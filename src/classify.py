import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from extract import extract_all_texts, extract_text

# -----------------------------
# Configs / Constantes
# -----------------------------
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

STOPWORDS_PT = [
    "a","à","às","ao","aos","as","o","os","de","do","da","dos","das","no","na","nos","nas",
    "um","uma","uns","umas","por","para","com","sem","sob","sobre","entre","até","após",
    "e","ou","mas","como","que","se","sua","seu","suas","seus","num","numa","numas","nuns",
    "é","ser","são","foi","eram","era","será","sido","tem","têm","ter","há","haver","está",
    "estão","estar","estavam","estive","estivemos","estiveram","em","àquele","àquela","àqueles","àquelas",
    "este","esta","estes","estas","isso","isto","aquele","aquela","aqueles","aquelas","qual","quais",
    "deve","devem","podem","pode","não","sim","também","muito","muitos","muita","muitas","menos","mais",
    "todo","toda","todos","todas","mesmo","mesma","mesmos","mesmas","cada","outro","outra","outros","outras",
    "seja","sejam","seriam","seria","deste","desta","destes","destas","neste","nesta","nestes","nestas"
]
# (opcional) garantir ordem estável:
STOPWORDS_PT = sorted(set(STOPWORDS_PT))

DEFAULT_VECTORIZER = TfidfVectorizer(
    lowercase=True,
    stop_words=STOPWORDS_PT,   # agora é list
    ngram_range=(1, 3),
    min_df=1
)

# -----------------------------
# Dataset
# -----------------------------
def load_labels(labels_path: Path) -> Dict[str, str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json não encontrado em: {labels_path}")
    return json.loads(labels_path.read_text(encoding="utf-8"))

def _filename_as_text(fname: str) -> str:
    # Usa underscores e números como tokens úteis; remove extensão
    base = fname.rsplit(".", 1)[0]
    # Troca underscores por espaço e põe o nome em caixa baixa
    return base.replace("_", " ").lower()

def build_dataset(
    textos: Dict[str, str],
    labels_map: Dict[str, str],
    include_filename_feature: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    Retorna (X_texts, y_labels, used_filenames) apenas dos PDFs que têm rótulo em labels_map.
    Se include_filename_feature=True, concatena o nome do arquivo ao início do texto.
    """
    X, y, used = [], [], []
    for fname, text in textos.items():
        if fname in labels_map:
            if include_filename_feature:
                text = _filename_as_text(fname) + "\n" + text
            X.append(text)
            y.append(labels_map[fname])
            used.append(fname)
    if not X:
        raise ValueError("Nenhum texto com rótulo correspondente. Verifique os nomes em labels.json.")
    return X, y, used

# -----------------------------
# Treino
# -----------------------------
def train_and_eval(
    X_texts: List[str],
    y_labels: List[str],
    model_type: str = "logreg",
    test_size: float = 0.3,
    random_state: int = 42
):
    """
    Treina o classificador e avalia (se possível). Retorna (vectorizer, label_encoder, model).
    model_type: 'logreg' ou 'nb'
    """
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    vectorizer = DEFAULT_VECTORIZER
    X = vectorizer.fit_transform(X_texts)

    # Tenta split estratificado; se não der (poucos exemplos), treina em 100% e avisa
    can_split = True
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        can_split = False
        X_train, y_train = X, y
        X_test = y_test = None

    # Modelo
    if model_type == "nb":
        model = MultinomialNB()
    else:
        # class_weight="balanced" ajuda em classes desbalanceadas
        model = LogisticRegression(max_iter=400, class_weight="balanced")

    model.fit(X_train, y_train)

    if can_split:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("\n== Avaliação ==")
        print(f"Acurácia: {acc:.4f}")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    else:
        print("\n[AVISO] Dataset muito pequeno para avaliação com split estratificado. Modelo treinado em 100% dos dados.")

    return vectorizer, le, model

# -----------------------------
# Persistência
# -----------------------------
def save_artifacts(vectorizer, label_encoder, model, prefix: str = "clf"):
    joblib.dump(vectorizer, MODELS_DIR / f"{prefix}_vectorizer.joblib")
    joblib.dump(label_encoder, MODELS_DIR / f"{prefix}_label_encoder.joblib")
    joblib.dump(model, MODELS_DIR / f"{prefix}_model.joblib")
    print(f"\nModelos salvos em: {MODELS_DIR}/ {prefix}_*.joblib")

def load_artifacts(prefix: str = "clf"):
    vectorizer = joblib.load(MODELS_DIR / f"{prefix}_vectorizer.joblib")
    label_encoder = joblib.load(MODELS_DIR / f"{prefix}_label_encoder.joblib")
    model = joblib.load(MODELS_DIR / f"{prefix}_model.joblib")
    return vectorizer, label_encoder, model

# -----------------------------
# Predição
# -----------------------------
def predict_text(text: str, vectorizer, label_encoder, model, return_proba: bool = True):
    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]
    label = label_encoder.inverse_transform([y_pred])[0]

    if return_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        return label, confidence
    else:
        return label, None

def predict_pdf_file(pdf_filename: str, vectorizer, label_encoder, model, prior_weight: float = 1.25):
    """
    Classifica um PDF usando texto + prior pelo nome do arquivo.
    prior_weight > 1 aumenta a prob. da classe encontrada no nome (ex.: 'Portaria_...')
    """
    base = _filename_as_text(pdf_filename)
    text = extract_text(pdf_filename)
    text = base + "\n" + text

    # vetoriza
    X = vectorizer.transform([text])

    # proba original do modelo
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]  # ndarray shape (n_classes,)
    else:
        # fallback se for um modelo sem proba
        y_pred = model.predict(X)[0]
        label = label_encoder.inverse_transform([y_pred])[0]
        return label, None

    classes = list(label_encoder.classes_)
    # identifica “hint” pelo nome
    hint = None
    if "lei" in base:
        hint = "Lei"
    elif "portaria" in base:
        hint = "Portaria"
    elif "resolucao" in base or "resolução" in base:
        hint = "Resolução"

    # aplica prior leve
    if hint in classes:
        idx = classes.index(hint)
        proba = proba.copy()
        proba[idx] *= prior_weight
        proba = proba / proba.sum()  # renormaliza

    # decide
    i = int(proba.argmax())
    label = classes[i]
    confidence = float(proba[i])
    THRESH = 0.45  # ajuste a gosto
    if confidence < THRESH:
        return "Desconhecido", confidence
    return label, confidence


def predict_all_in_data(vectorizer, label_encoder, model, save_csv: bool = True):
    data_dir = Path(__file__).parent.parent / "data"
    pdfs = sorted(list(data_dir.glob("*.pdf")))
    results = []

    for pdf in pdfs:
        try:
            label, conf = predict_pdf_file(pdf.name, vectorizer, label_encoder, model)
            results.append((pdf.name, label, conf))
            conf_str = f" ({conf:.2f})" if conf is not None else ""
            print(f"→ {pdf.name}: {label}{conf_str}")
        except Exception as e:
            print(f"[ERRO] {pdf.name}: {e}")

    if save_csv and results:
        import csv
        out_csv = Path(__file__).parent.parent / "predicoes.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arquivo", "classe", "confianca"])
            for r in results:
                w.writerow([r[0], r[1], f"{r[2]:.4f}" if r[2] is not None else ""])
        print(f"\nPredições salvas em: {out_csv}")

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Classificação de PDFs por ML (TF-IDF + LogReg/NB) com melhorias")
    parser.add_argument("--labels", type=str, default=str(Path(__file__).parent.parent / "labels.json"),
                        help="Caminho para labels.json (arquivo→classe)")
    parser.add_argument("--model", type=str, choices=["logreg", "nb"], default="logreg",
                        help="Tipo de modelo: Logistic Regression (logreg) ou Naive Bayes (nb)")
    parser.add_argument("--prefix", type=str, default="clf", help="Prefixo dos artefatos salvos em /models")
    parser.add_argument("--train", action="store_true", help="Treinar (e avaliar) o modelo")
    parser.add_argument("--predict-all", action="store_true", help="Classificar todos os PDFs em /data")
    parser.add_argument("--predict-one", type=str, help="Classificar um PDF específico (nome do arquivo em /data)")
    args = parser.parse_args()

    labels_path = Path(args.labels)

    if args.train:
        print("== Extraindo textos de /data ==")
        textos = extract_all_texts()
        print("== Carregando rótulos ==")
        labels_map = load_labels(labels_path)

        X_texts, y_labels, used = build_dataset(textos, labels_map, include_filename_feature=True)
        print(f"Treinando com {len(X_texts)} documentos rotulados: {used}")

        vectorizer, le, model = train_and_eval(
            X_texts, y_labels, model_type=args.model
        )
        save_artifacts(vectorizer, le, model, prefix=args.prefix)

    if args.predict_all or args.predict_one:
        vectorizer, le, model = load_artifacts(prefix=args.prefix)

    if args.predict_all:
        predict_all_in_data(vectorizer, le, model, save_csv=True)

    if args.predict_one:
        label, conf = predict_pdf_file(args.predict_one, vectorizer, le, model)
        conf_str = f" ({conf:.2f})" if conf is not None else ""
        print(f"{args.predict_one}: {label}{conf_str}")

if __name__ == "__main__":
    main()
