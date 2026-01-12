import json
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

K_PADRAO = 5

# mesma coisa de sempre  pra saber o diretorio e onde 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

BULKS = {
    "abacaxi": os.path.join(PROJECT_DIR, "bulk", "bulk-abacaxi.json"),
}

INDICES_DIR = os.path.join(PROJECT_DIR, "indices")


MODEL_NAME = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir"

# onde salvar/ler artefatos prontos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDICES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "indices"))

def _path_index(cultura: str) -> str:
    return os.path.join(INDICES_DIR, f"{cultura}.faiss")

def _path_docs(cultura: str) -> str:
    return os.path.join(INDICES_DIR, f"{cultura}_docs.json")

def _path_medias() -> str:
    return os.path.join(INDICES_DIR, "medias_cultura.json")


class QA_Doc(BaseModel):
    text: str
    index: str | None = None
    question_number: int | None = None
    question: str | None = None
    answer: str | None = None
    chapter: str | None = None
    book: str | None = None
    book_id: str | None = None
    epub: str | None = None
    pdf: str | None = None
    html: str | None = None
    year: str | None = None
    embedding: list[float] | None = None


# ------------------------ LEITURA DO BULK (SÓ NO BUILD) ------------------------ #

def carregar_doc(jsonl_path: str):
    docs = []

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError("Arquivo não encontrado: " + jsonl_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        linhas = [ln.strip() for ln in f.readlines() if ln.strip()]

    i = 0
    while i < len(linhas):
        index_line = json.loads(linhas[i])
        data_line = json.loads(linhas[i + 1])

        question = data_line.get("question", "")
        answer = data_line.get("answer", "")

        doc = QA_Doc(
            text=(question + "\n" + answer).strip(),
            index=index_line["index"]["_id"],
            question_number=data_line.get("question_number"),
            question=question,
            answer=answer,
            chapter=data_line.get("chapter"),
            book=data_line.get("book"),
            book_id=data_line.get("book_id"),
            epub=data_line.get("epub"),
            pdf=data_line.get("pdf"),
            year=str(data_line.get("year")) if data_line.get("year") else ""
        )
        docs.append(doc)
        i += 2
    return docs


# ------------------------ EMBEDDINGS ------------------------ #

def gerar_embeddings(textos, model):
    emb = model.encode(textos, convert_to_numpy=True, show_progress_bar=False) # transforma os textos em Ids ( tokerização) + pooling
    emb = emb.astype("float32") # conversao
    normas = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12 # normaliza
    emb = emb / normas
    return emb


# ------------------------ DETECÇÃO DE CULTURA (RUNTIME) ------------------------ #

def detectar_cultura(query: str, sistema: Dict[str, Any]) -> str:
    # se só tiver uma cultura, evita custo e ambiguidade
    culturas = list(sistema["indices"].keys())
    if len(culturas) == 1:
        return culturas[0]

    model = sistema["model"]
    medias_cultura = sistema.get("medias_cultura", {})

    emb_query = gerar_embeddings([query], model)
    if emb_query.shape[0] == 0 or not medias_cultura:
        return culturas[0]

    vetor_query = emb_query[0]

    melhor_cultura = culturas[0]
    melhor_similaridade = -1e9

    for cultura, media_vec in medias_cultura.items():
        sim = float(np.dot(vetor_query, media_vec))
        if sim > melhor_similaridade:
            melhor_similaridade = sim
            melhor_cultura = cultura

    return melhor_cultura


# ======================================================================
# 1) BUILD OFFLINE: gera e salva índices/metadata (roda 1x quando atualizar)
# ======================================================================

def build_indices() -> None:
    os.makedirs(INDICES_DIR, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    print(f"[build] Modelo carregado: {MODEL_NAME}")

    medias_cultura: Dict[str, list] = {}

    for cultura, caminho_bulk in BULKS.items():
        print(f"[build] Lendo bulk '{cultura}' em {caminho_bulk}...")

        docs = carregar_doc(caminho_bulk)
        if not docs:
            print(f"[build] Nenhum doc para '{cultura}'. Pulando.")
            continue

        textos = [d.text for d in docs]
        emb = gerar_embeddings(textos, model)
        dim = emb.shape[1]

        index_flat = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(index_flat)

        # Cria IDs para os documentos
        ids = np.arange(len(docs), dtype="int64")
        index.add_with_ids(emb, ids)

        # salva índice
        caminho_index = _path_index(cultura)
        faiss.write_index(index, caminho_index)
        print(f"[build] Índice salvo em: {caminho_index}")


        # salva docs (metadados) em json 
        docs_out = {}
        
        for i, doc in enumerate(docs):
            docs_out[str(i)] = {
                "index": doc.index,
                "question_number": doc.question_number,
                "question": doc.question,
                "answer": doc.answer,
                "chapter": doc.chapter,
                "book": doc.book,
                "book_id": doc.book_id,
                "pdf": doc.pdf,
                "epub": doc.epub,
                "year": doc.year,
            }

        with open(_path_docs(cultura), "w", encoding="utf-8") as f:
            json.dump(docs_out, f, ensure_ascii=False)
        print(f"[build] Salvo docs em: {_path_docs(cultura)}")

        # média por cultura (normalizada) 
        media = emb.mean(axis=0)
        norma = np.linalg.norm(media)
        media_normalizada = media / (norma + 1e-12)
        medias_cultura[cultura] = media.astype("float32").tolist()

        print(f"[build] Cultura '{cultura}': {len(docs)} docs, dim={dim}")

    with open(_path_medias(), "w", encoding="utf-8") as f:
        json.dump(medias_cultura, f, ensure_ascii=False, indent=2)

    print(f"[build] Salvo medias_cultura em: {_path_medias()}")
    print("[build] Finalizado.")


# ======================================================================
# 2) RUNTIME (API): só carrega artefatos prontos e faz busca
# ======================================================================

def inicializar_sistema() -> Dict[str, Any]:
    model = SentenceTransformer(MODEL_NAME)
    print(f"[pipeline] Modelo carregado: {MODEL_NAME}")

    indices_faiss: Dict[str, faiss.Index] = {}
    docs_por_cultura: Dict[str, Dict[str, dict]] = {}
    medias_cultura_np: Dict[str, np.ndarray] = {}

    # carrega medias_cultura (se existir)
    if os.path.exists(_path_medias()):
        with open(_path_medias(), "r", encoding="utf-8") as f:
            medias_json = json.load(f)
        for cultura, vec_list in medias_json.items():
            medias_cultura_np[cultura] = np.array(vec_list, dtype="float32")

    for cultura in BULKS.keys():
        p_index = _path_index(cultura)
        p_docs = _path_docs(cultura)

        if not os.path.exists(p_index) or not os.path.exists(p_docs):
            raise RuntimeError(
                f"Artefatos não encontrados para '{cultura}'. "
                f"Rode: python pipeline.py --build. Faltando: {p_index} ou {p_docs}"
            )

        index = faiss.read_index(p_index)
        indices_faiss[cultura] = index

        with open(p_docs, "r", encoding="utf-8") as f:
            docs_por_cultura[cultura] = json.load(f)

        print(f"[pipeline] Cultura '{cultura}': índice+docs carregados.")

    sistema = {
        "model": model,
        "indices": indices_faiss,
        "docs": docs_por_cultura,         # dict[str(cultura)][str(id_doc)] -> dict metadata
        "medias_cultura": medias_cultura_np,
        "k_padrao": K_PADRAO,
    }

    print("[pipeline] Sistema inicializado com sucesso.")
    return sistema

    #coração do retrivel  
def buscar_top_k(sistema, query, k=5, cultura: Optional[str] = None):
    if query is None: # valido meu query isso faz evitar erro e fazer busca sem sentido
        return []
    query = str(query).strip()
    if query == "":
        return []

    if k is None or k <= 0:   #valido meu k tbm
        k = sistema["k_padrao"]

    model = sistema["model"]
    indices = sistema["indices"]
    docs_por_cultura = sistema["docs"]

    if cultura is not None:  # escolho minha cultura  
        cultura = str(cultura).strip() or None

    if cultura is None or cultura not in indices:
        cultura_usada = detectar_cultura(query, sistema)
    else: # se ela nao veio ou e invalida esclho automaticamente
        cultura_usada = cultura

    index = indices.get(cultura_usada)
    docs_dict = docs_por_cultura.get(cultura_usada)
    if index is None or docs_dict is None: # faz a verificação se  nenhuma existir  vai pro return
        return []  

    emb_query = gerar_embeddings([query], model)  # gera uma matriz [1,dim]
    if emb_query.shape[0] == 0:     
        return []

    D, I = index.search(emb_query, k)  # busca no faiss atraves dos I(indices )  D(distancia)

     #Montagem do resultado com metadados + rank
    resultados = []
    ids = I[0]
    distancias = D[0]

    rank = 1
    for j in range(len(ids)):
        id_doc = int(ids[j])
        dist = float(distancias[j])
        if id_doc < 0:
            continue

        doc_meta = docs_dict.get(str(id_doc)) # recupero meus metados
        if doc_meta is None:
            continue
 # monto minha configuração final meu objeto neste formato 
        resultados.append({
            "id_doc": id_doc,
            "index": doc_meta.get("index"),
            "question_number": doc_meta.get("question_number"),
            "question": doc_meta.get("question"),
            "answer": doc_meta.get("answer"),
            "chapter": doc_meta.get("chapter"),
            "book": doc_meta.get("book"),
            "book_id": doc_meta.get("book_id"),
            "pdf": doc_meta.get("pdf"),
            "epub": doc_meta.get("epub"),
            "year": doc_meta.get("year"),
            "score": dist,          
            "rank": rank,
            "cultura": cultura_usada,
        })
        rank += 1

    return resultados  # conteúdo + metadados + score + rank   fim do retrivel
  

# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        build_indices()
        raise SystemExit(0)

    # modo default: runtime (como seria na API)
    sistema = inicializar_sistema()
    print("Culturas:", list(sistema["indices"].keys()))

    resultados = buscar_top_k(sistema, query="como plantar abacaxi?", k=5, cultura=None)

    print("\nResultados:")
    for r in resultados:
        print(r["rank"], r["id_doc"], r["score"])
        print("P:", r["question"])
        print("R:", r["answer"])
        print()
