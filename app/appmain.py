from fastapi import FastAPI, HTTPException, status, Security
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
import re
import requests
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------- ENV + PATHS ---------------------------- #

BASE_DIR = Path(__file__).resolve().parent    
PROJECT_ROOT = BASE_DIR.parent                      
ENV_PATH = PROJECT_ROOT / ".env"

# garante  o import
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=ENV_PATH, override=True)

API_KEY = os.getenv("API_KEY") #le a variavel ambiete 
if not API_KEY: # caso nao tenha pare
    raise RuntimeError(f"API_KEY não encontrada em {ENV_PATH}")  

API_KEYS = {API_KEY}  # Cria um conjunto com a chave válida

# IMPORTA DO PIPELINE 
try:
    from core.pipeline import inicializar_sistema_runtime as inicializar_sistema # importa a versão runtime do pipeline
except Exception: # nao deu entra  no fallback
    from core.pipeline import inicializar_sistema  # fallback caso tenha uma falha

from core.pipeline import buscar_top_k

# ---------------------------- Autentifica ---------------------------- #

# query string header HTTP

api_key_query = APIKeyQuery(name="api-key", auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# chamando a api 
def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
):
    if api_key_query in API_KEYS:
        return api_key_query
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )

# ---------------------------- FASTAPI APP ---------------------------- #

app = FastAPI(title="Chatbot Abacaxi API", debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------- ESTADO GLOBAL ---------------------------- #

SISTEMA_RAG = None

@app.on_event("startup")
def startup_event():
    global SISTEMA_RAG
    SISTEMA_RAG = inicializar_sistema()
    if SISTEMA_RAG is None:
        raise RuntimeError("Falha ao inicializar o sistema RAG.")

# ---------------------------- MODELOS ---------------------------- #

# define a estrutura esperada de cada uma 
class ChatRequest(BaseModel):
    query: str
    k: int = 5
    cultura: Optional[str] = None

class ChatResult(BaseModel):
    id_doc: str
    index: str
    question: str
    answer: str
    rank: int
    cultura: str

class ChatResponse(BaseModel):
    query: str
    k: int
    cultura_usada: Optional[str]
    resultados: List[ChatResult]

class AnswerRequest(BaseModel):
    query: str
    k: int = 5
    cultura: Optional[str] = None

class SourceItem(BaseModel):
    index: str
    pergunta: str

class AnswerResponse(BaseModel):
    query: str
    answer: str
    cultura_usada: Optional[str]
    sources: List[SourceItem]

# ---------------------------- UTILS ---------------------------- #

def chamar_ollama_chat(messages: list, model: str = "qwen3:8b") -> str:
    url = os.getenv("OLLAMA_URL", "https://ollama.tieta.eu:8443/api/chat")

    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()

    raw = (r.json().get("message", {}).get("content") or "").strip()

    # 1) remove <think>...</think>
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2) se vier HTML (<p>...), corta tudo antes do primeiro <p>
    ppos = clean.lower().find("<p")
    if ppos != -1:
        clean = clean[ppos:].strip()

    # 3) remove "Fontes usadas:" e tudo depois (
    clean = re.split(r"\n\s*Fontes usadas\s*:", clean, flags=re.IGNORECASE)[0].strip()

    # 4) remove HTML para virar texto "chat"
    clean = re.sub(r"<[^>]+>", "", clean).strip()

    return clean


# ---------------------------- ENDPOINTS ---------------------------- #

@app.get("/health")
def health():
    return {"status": "ok", "sistema_carregado": SISTEMA_RAG is not None}

@app.post("/chatbot/query", response_model=ChatResponse)
def consultar_chatbot(req: ChatRequest, api_key: str = Security(get_api_key)):
    if SISTEMA_RAG is None:
        raise HTTPException(status_code=500, detail="Sistema não inicializado")

        
    # parte do meu retrivel  chamando a funçao
    resultados_brutos = buscar_top_k(
        SISTEMA_RAG, query=req.query, k=req.k, cultura=req.cultura
    ) or []

    resultados_model: List[ChatResult] = []
    for idx, r in enumerate(resultados_brutos, start=1):
        resultados_model.append(
            ChatResult(
                id_doc=str(r.get("id_doc", "")),
                index=str(r.get("index", "")),
                question=str(r.get("question", "") or ""),
                answer=str(r.get("answer", "") or ""),
                rank=int(r.get("rank", idx)),
                cultura=str(r.get("cultura", "")),
            )
        )

    cultura_usada = req.cultura or (resultados_brutos[0].get("cultura") if resultados_brutos else None)

    return ChatResponse(
        query=req.query,
        k=req.k,
        cultura_usada=cultura_usada,
        resultados=resultados_model,
    )

@app.post("/chatbot/answer", response_model=AnswerResponse)
def responder_chatbot(req: AnswerRequest, api_key: str = Security(get_api_key)):
    if SISTEMA_RAG is None:
        raise HTTPException(status_code=500, detail="Sistema não inicializado")

    resultados = buscar_top_k(
        SISTEMA_RAG, query=req.query, k=req.k, cultura=req.cultura
    ) or []

    if not resultados:
        return AnswerResponse(
            query=req.query,
            answer="Não encontrei informação suficiente no material disponível.",
            cultura_usada=req.cultura,
            sources=[],
        )

    cultura_usada = req.cultura or resultados[0].get("cultura")

    # fontes (para retornar no JSON)
    sources: List[SourceItem] = []
    for r in resultados:
        sources.append(SourceItem(
            index=str(r.get("index", "")),
            pergunta=str((r.get("question") or "")).strip()
        ))

    # ----------------------------
    # MODELO MEMÓRIA
    # ----------------------------
    memoria_blocos = []
    for i, r in enumerate(resultados, start=1):
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a:
            continue
        memoria_blocos.append(f"[{i}] Q: {q}\n    A: {a}")

    memoria_txt = "\n\n".join(memoria_blocos)
   # Isso é o prompt  que será levado ao modelo
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente técnico agrícola. "
                "Responda APENAS em português. "
                "Use a MEMÓRIA apenas como base factual. "
                "Produza UMA resposta final, objetiva e sintética, "
                "combinando as informações mais relevantes da memória. "
                "NÃO mostre raciocínio interno, análises, explicações do processo "
                "nem tags como <think>. "
                "Se a memória não contiver informação suficiente, diga isso explicitamente."
            )
        },
        {
            "role": "user",  # prompt do divo RAg
            "content": (
                f"MEMÓRIA (top-{len(memoria_blocos)} por similaridade):\n"
                f"{memoria_txt}\n\n"
                f"PERGUNTA FINAL:\n{req.query}"
            )
        }
    ]

    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")  
    answer = chamar_ollama_chat(messages, model=model)

    # fallback se o LLM retornar vazio
    if not answer:
        answer = (resultados[0].get("answer") or "").strip() or \
                 "Recuperei documentos relevantes, mas não consegui gerar uma resposta."

    return AnswerResponse(
        query=req.query,
        answer=answer,
        cultura_usada=cultura_usada,
        sources=sources,
    )
