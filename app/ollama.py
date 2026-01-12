import requests
import re

URL = "https://ollama.tieta.eu:8443/api/chat"
MODEL = "qwen3:8b"  

# --------- MEMÓRIA (simulação do top-k do RAG) ----------
memoria_itens = [
    (
        "Como é possível reconhecer mudas, plantas e frutos infectados pelo agente causal da fusariose?",
        "Os sintomas são caracterizados principalmente pela exsudação de resina incolor pelos tecidos atacados, "
        "que ao contato com o ar solidifica e forma massas irregulares marrom-escuras."
    ),
    (
        "Os sintomas da murcha-do-abacaxi podem ser confundidos com os provocados por outros problemas?",
        "Sim. Podem ser confundidos com problemas como Phytophthora, deficiência de cobre e nematoides, "
        "sendo necessária avaliação conjunta dos sintomas."
    ),
    (
        "Quando o abacaxi deve ser colhido?",
        "O abacaxi deve ser colhido no estágio “de vez”, quando a casca passa de verde-escura para verde-clara "
        "com início de amarelecimento."
    ),
]

pergunta_final = "Quais são os sintomas de fusariose no abacaxi?"

# --------- CONSTRÓI A MEMÓRIA EM STRING ----------
blocos = []
for i, (q, a) in enumerate(memoria_itens, start=1):
    blocos.append(f"[{i}] Q: {q}\n    A: {a}")

memoria_txt = "\n\n".join(blocos)

# --------- MESSAGES (MODELO MEMÓRIA)  #prompt restrições fortes) ( retrivel)
messages = [
    {
        "role": "system",
        "content": (
            "Você é um assistente técnico agrícola. "
            "Sua tarefa é gerar UMA resposta final, clara e objetiva. "
            "Responda exclusivamente em português (pt-BR). "
            "Use a MEMÓRIA apenas como base factual. "
            "Sintetize a melhor resposta possível a partir dela. "
            "NÃO inclua raciocínio interno, explicações do processo, listas de fontes "
            "ou qualquer texto além da resposta final."
        )
    },
    {
        "role": "user",   #User prompt (memória + pergunta)
        "content": (
            f"MEMÓRIA (top-{len(memoria_itens)} por similaridade):\n"
            f"{memoria_txt}\n\n"
            f"PERGUNTA FINAL:\n{pergunta_final}"
        )
    }
]     
# chama a infra da tieta
payload = {
    "model": MODEL,
    "messages": messages,
    "stream": False
}

r = requests.post(URL, json=payload, timeout=120)
r.raise_for_status()

raw = r.json().get("message", {}).get("content", "") or ""

# --------- PÓS-PROCESSAMENTO (blindagem) ----------
# remove <think>...</think>
clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)

# remove HTML se vier (<p>, etc.)
clean = re.sub(r"<[^>]+>", "", clean)

# remove qualquer lixo antes da resposta
clean = clean.strip()

print(clean)
