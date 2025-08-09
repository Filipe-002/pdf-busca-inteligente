from pathlib import Path
import pymupdf as fitz

# lê os pdfs da pasta data e extrai os textos
def extract_text(pdf_filename):
    pdf_path = Path(__file__).parent.parent / "data" / pdf_filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
    
    doc = fitz.open(pdf_path)

    text = '\n'.join(page.get_text() for page in doc)
    
    output_file = Path(__file__).parent.parent / "extracted_data" / f"{pdf_filename}.txt"
    output_file.write_text(text, encoding="utf-8")

    print(f"Texto extraído e salvo em: {output_file}")

    return text

# percorre todos os pdfs, chamando extract_text 
def extract_all_texts(save_txt=False):
    data_dir = Path(__file__).parent.parent / "data"
    pdf_files = list(data_dir.glob("*.pdf"))
    resultados = {}
    for pdf_file in pdf_files:
        try:
            print(f"Extraindo: {pdf_file.name}")
            texto = extract_text(pdf_file.name)  
            if save_txt:
                output_file = Path(__file__).parent / f"{pdf_file.name}.txt"
                output_file.write_text(texto, encoding="utf-8")
            resultados[pdf_file.name] = texto
        except Exception as e:
            print(f"Erro em {pdf_file.name}: {e}")
    return resultados

if __name__ == "__main__":
    resultados = extract_all_texts()

    print("\nResumo:")
    for nome, conteudo in resultados.items():
        print(f"→ {nome}: {len(conteudo)} caracteres extraídos")