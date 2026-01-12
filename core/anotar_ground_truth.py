import json
import os
import re
import time
from typing import List, Dict, Any

import requests
from pipeline import inicializar_sistema, buscar_top_k

TOPN_DENSE = 30
MAX_RELEVANTES = 8
SLEEP_BETWEEN_CALLS = 0.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

INPUT_QUERIES_PATH = os.path.join(PROJECT_DIR, "dados", "queries_para_anotacao.json")
OUTPUT_GT_PATH = os.path.join(PROJECT_DIR, "dados", "recallak.json")
OUTPUT_DEBUG_PATH = os.path.join(PROJECT_DIR, "dados", "recallak_debug.json")

USE_LLM_JUDGE = True

# IMPORTANTE: aqui é /api/chat (messages), não /api/generate (prompt)
OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ollama.tieta.eu:8443/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")


def limpar_html(s: str) -> str:
    s = re.sub(r"<table[\s\S]*?</table>", " [TABELA] ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_doc_id(r: Dict[str, Any]) -> str:
    """
    Retorna o identificador canônico do documento/chunk.
    IMPORTANTÍSSIMO: esse ID precisa ser o mesmo usado depois no cálculo do recall@k.
    """
    for k in ("id", "doc_id", "chunk_id", "source_id", "document_id", "index"):
        v = r.get(k)
        if v is None:
            continue
        v = str(v).strip()
        if v:
            return v
    return ""


def extrair_trecho_relevante(query: str, text: str, limit: int = 2500) -> str:
    clean = limpar_html(text)
    q = query.lower()

    # Heurística para queries de espaçamento: recorta perto de medidas
    if "espaç" in q:
        patterns = [
            r"\b\d+([,\.]\d+)?\s*x\s*\d+([,\.]\d+)?(\s*x\s*\d+([,\.]\d+)?)?\b",
            r"\b\d+\s*cm\b",
            r"\b\d+([,\.]\d+)?\s*m\b",
        ]
        low = clean.lower()
        for p in patterns:
            m = re.search(p, low)
            if m:
                start = max(0, m.start() - 500)
                end = min(len(clean), m.end() + 800)
                return clean[start:end][:limit]

    return clean[:limit]


def filtro_por_regra(query: str, candidatos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q = query.lower()

    if "espaç" in q:
        patterns = [
            r"espaç",
            r"distân",
            r"entre\s+fil",
            r"entre\s+plant",
            r"\b\d+([,\.]\d+)?\s*x\s*\d+([,\.]\d+)?(\s*x\s*\d+([,\.]\d+)?)?\b",
            r"\b\d+\s*cm\b",
            r"\b\d+([,\.]\d+)?\s*m\b",
        ]

        out = []
        for r in candidatos:
            text = (r.get("question") or "") + " " + (r.get("answer") or "")
            text = limpar_html(text).lower()
            if any(re.search(p, text) for p in patterns):
                out.append(r)
        return out

    return candidatos


def judge_llm(query: str, candidatos: List[Dict[str, Any]]) -> List[str]:
    itens = []
    for r in candidatos:
        doc_id = get_doc_id(r)
        if not doc_id:
            continue
        itens.append({
            "doc_id": doc_id,
            "question": (r.get("question") or "").strip(),
            "answer": extrair_trecho_relevante(query, (r.get("answer") or ""), limit=2500),
        })

    if not itens:
        return []
    # prompt controle do comportamento
    system = (
        "Você é um avaliador de relevância para RAG.\n"
        "Marque como RELEVANTE somente itens que respondem diretamente à query.\n"
        "Se a query pedir parâmetro numérico (ex.: espaçamento), só marque itens com valores explícitos.\n"
        f"Retorne no máximo {MAX_RELEVANTES} doc_id.\n"
        "Responda APENAS com um bloco <json>...</json> contendo JSON válido.\n"
        "NÃO escreva nada fora do <json>.\n"
        "Formato: <json>{\"relevantes\": [\"doc_id\", ...]}</json>\n"
    
    )

   #prompt  Aqui você passa a query e os documentos candidatos (recortados)
    user_content = (
        "ENTRADA (JSON):\n"
        f"{json.dumps({'query': query, 'candidates': itens}, ensure_ascii=False)}\n\n"
        "RETORNE SOMENTE:\n"
        "<json>{\"relevantes\": []}</json>"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    raw = (resp.json().get("message", {}) or {}).get("content", "") or ""

    # blindagem
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", "", raw).strip()

    # parse: preferir <json>, senão fallback { ... }
    m = re.search(r"<json>\s*([\s\S]*?)\s*</json>", raw, flags=re.I)
    if m:
        raw_json = m.group(1).strip()
    else:
        m2 = re.search(r"\{[\s\S]*\}", raw)
        if not m2:
            return []
        raw_json = m2.group(0).strip()

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    relevantes = data.get("relevantes", [])
    if not isinstance(relevantes, list):
        return []

    out = []
    seen = set()
    for doc_id in relevantes:
        if not isinstance(doc_id, str):
            continue
        doc_id = doc_id.strip()
        if doc_id and doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)
        if len(out) >= MAX_RELEVANTES:
            break

    return out


def main():
    sistema = inicializar_sistema()

    with open(INPUT_QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    ground_truth = []
    debug_rows = []

    for q in queries:
        query = str(q).strip()
        if not query:
            continue

        candidatos = buscar_top_k(sistema, query=query, k=TOPN_DENSE, cultura=None) or []

        if not candidatos:
            ground_truth.append({"query": query, "relevantes": []})
            debug_rows.append({
                "query": query,
                "top10_ids": [],
                "filtrados_top20_ids": [],
                "relevantes": [],
                "usou_llm": bool(USE_LLM_JUDGE),
                "n_candidatos": 0,
                "n_filtrados": 0,
            })
            continue

        top_ids = [get_doc_id(r) for r in candidatos[:10]]

        candidatos_filtrados = filtro_por_regra(query, candidatos)
        filt_ids = [get_doc_id(r) for r in candidatos_filtrados[:20]]

        if USE_LLM_JUDGE:
            relevantes = judge_llm(query, candidatos_filtrados)
        else:
            relevantes = []
            for r in candidatos_filtrados:
                did = get_doc_id(r)
                if did:
                    relevantes.append(did)
                if len(relevantes) >= MAX_RELEVANTES:
                    break

        # fallback para não deixar ground truth vazio
        if not relevantes:
            base = candidatos_filtrados if candidatos_filtrados else candidatos
            relevantes = []
            for r in base[:3]:
                did = get_doc_id(r)
                if did:
                    relevantes.append(did)

        ground_truth.append({"query": query, "relevantes": relevantes})

        debug_rows.append({
            "query": query,
            "top10_ids": top_ids,
            "filtrados_top20_ids": filt_ids,
            "relevantes": relevantes,
            "usou_llm": bool(USE_LLM_JUDGE),
            "n_candidatos": len(candidatos),
            "n_filtrados": len(candidatos_filtrados),
        })

        if SLEEP_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)

    with open(OUTPUT_GT_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DEBUG_PATH, "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, ensure_ascii=False, indent=2)

    print("recallak.json gerado em:", OUTPUT_GT_PATH)
    print("debug gerado em:", OUTPUT_DEBUG_PATH)


if __name__ == "__main__":
    main()
