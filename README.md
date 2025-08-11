# PDFScanner — Extração, Classificação, Busca e Chat (RAG) em PDFs

Estudo de caso em Python, em que:

1. **Extraímos texto** dos PDFs (PyMuPDF).  
2. **Classificanos** automaticamente (`Lei`, `Portaria`, `Resolução`) com **TF-IDF + Naive Bayes**, algoritmos de aprendizado de máquina (Scikit-Learn).  
3. **Buscamos por linguagem natural** com **BM25 (NLTK)**, retornando os documentos mais relevantes.  
4. **Chat com LLM local** via **Ollama** (RAG): a resposta é gerada **apenas** com base nos trechos recuperados.

---

## Requisitos

- **Python 3.11+**  
- **Ollama** instalado e em execução (`ollama serve`)  
- Um modelo local disponível no Ollama de sua preferência (ex.: `deepseek-r1:1.5b`, `llama3.1:8b`, etc.)  
- Windows, macOS ou Linux (sem GPU)

## 📂 Estrutura do projeto

📁 **pdf-busca-inteligente**  
├── 📂 **data/** — Coloque aqui os PDFs  
├── 📂 **extracted_data/** — Textos extraídos (.txt)  
├── 📂 **models/** — Vetorizador, label encoder e classificador  
├── 📂 **nltk_index/** — Índice BM25 persistido  
├── 📂 **src/**  
│   ├── 📄 **extract.py** — Extração de texto (PyMuPDF)  
│   ├── 📄 **classify.py** — Treino/predição (TF-IDF + Naive Bayes)  
│   ├── 📄 **search.py** — Índice BM25 (NLTK) + snippets  
│   ├── 📄 **chatnltk.py** — CLI de chat RAG com Ollama  
│   ├── 📄 **app.py** — Interface web (Streamlit)  
├── 📄 **labels.json** — Rótulos para treino supervisionado  
├── 📄 **requirements.txt** — Dependências do projeto  
└── 📄 **README.md** — Documentação do projeto


## Instalação (passo a passo)

### 1. Clone e crie seu ambiente virtual
```bash
git clone https://github.com/Filipe-002/pdf-busca-inteligente
cd pdf-busca-inteligente

# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate 
```
### 2. Instalação das dependências

```bash
pip install -r requirements.txt
```
### 3. Baixe os pacotes do NLTK

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('rslp')"
```

### 4. Coloque os PDFs em ./data (caso queira adicionar novos para testar)

```bash
Lei_9394_20121996.pdf
Portaria_44_28052025.pdf
```

## Feito tudo isso, para testar o projeto:

### 1. Extraia os textos:

```bash
python src/extract.py
```

### 2. Para fazer as classificações dos PDFs:

```bash
python src/classify.py --train --labels labels.json
```

### 3. Predizer todos os PDFs:

```bash
python src/classify.py --predict-all
```

### 4. Construa o índice BM25:

```bash
python src/search.py --build
```

### Pronto, já pode consultar no terminal:

```bash
python src/search.py --query "diretrizes da educação básica" --top 5
```

### Para instalar o modelo de IA:

```bash
ollama serve
ollama list
ollama pull <escolha o modelo que preferir>
```

### Para rodar a interface web e fazer perguntas ao chat:

```bash
streamlit run src/app_streamlit.py
```

O modelo padrão é definido em MODEL_NAME no app_streamlit.py.

Altere se necessário para um modelo disponível no seu Ollama.