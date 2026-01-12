from pipeline import inicializar_sistema, buscar_top_k

sistema = inicializar_sistema()

query = "como plantar abacaxi?"
top_n = 10

cands = buscar_top_k(sistema, query=query, k=top_n, cultura=None) or []

print("QUERY:", query)
print("=" * 100)

for i, r in enumerate(cands, start=1):
    doc_id = r.get("id") or r.get("doc_id") or r.get("index")
    print(f"\nRANK {i} | ID: {doc_id}")
    print("-" * 100)
    print("PERGUNTA:", r.get("question", "")[:300])
    print("RESPOSTA:", r.get("answer", "")[:1200])
