import json
import os
from typing import Dict, List, Any, Tuple
from pipeline import inicializar_sistema, buscar_top_k



K_LIST = [1, 3, 5, 10]

# Se seu recallak.json estiver em outro lugar, ajuste aqui.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../core
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # .../sadai-chat

GROUND_TRUTH_PATH = os.path.join(PROJECT_DIR, "dados", "recallak.json")

# -----------------------------
# Métricas
# -----------------------------
def recall_at_k(relevantes: List[str], recuperados: List[str], k: int) -> float:  #  relevantes: Lista de IDs/ (ground truth) k: Número de resultados a considerar no topo da lista
    """Recall@k = |relevantes ∩ topk| / |relevantes|""" # função explicita de como  e o recall numoero de relevantes  nos k primeiro dividido pelo total de numero relevantes
    if not relevantes:  # se nao tiver relevantes retorna zero
        return 0.0
    topk = recuperados[:k]
    acertos = len(set(relevantes) & set(topk))    #  Transforma ambas listas em conjuntos  ffaz a Interseção  Contagem: Calcula quantos relevantes foram efetivamente recuperados
    return acertos / float(len(set(relevantes)))  # calculo final  normaliza Divide o número de acertos pelo total de relevantes


# def mrr_at_k(relevantes: List[str], recuperados: List[str], k: int) -> float:
#     """
#     MRR@k (Mean Reciprocal Rank):
#     retorna 1/rank do primeiro relevante encontrado no top-k, senão 0.
#     """
#     if not relevantes:
#         return 0.0
#     relevantes_set = set(relevantes)
#     topk = recuperados[:k]
#     for i, doc_id in enumerate(topk, start=1):
#         if doc_id in relevantes_set:
#             return 1.0 / float(i)
#     return 0.0



# -----------------------------
# IO
# -----------------------------
def carregar_ground_truth(path: str) -> List[Dict[str, Any]]: # define a funçap 
    if not os.path.exists(path): # verifica se tem algo nesse caminho pasta/ arquivo
        raise FileNotFoundError(f"Ground truth não encontrado em: {path}") # caso nao tenha nada imterrompe a execução da função e lança uma exceção.

    with open(path, "r", encoding="utf-8") as f:  # abre o arquivo para ler
        data = json.load(f) # le  o arquivo parseia como json se o mesmo for valido vira um objeto Python

    if not isinstance(data, list): # é uma lista 
        raise ValueError("O ground truth deve ser uma LISTA de objetos JSON.") # nao e Interrompe a execução

    # Validação mínima
    for i, item in enumerate(data):
        if "query" not in item: # verifica se tem  a chave query no dicionario se não existir, o item não tem a pergunta que será usada no retrieval.
            raise ValueError(f"Item {i} sem campo 'query'.")   #  interrompe a execução imediatamente. 
        if "relevantes" not in item:   
            raise ValueError(f"Item {i} sem campo 'relevantes'.")  #  Verifica se existe a chave "relevantes" no item.           
        if not isinstance(item["relevantes"], list):  #Confere se o valor associado à chave "relevantes" é uma lista Python
            raise ValueError(f"Item {i}: 'relevantes' deve ser LISTA.") #Se relevantes não for lista função lança erro.
    return data


# -----------------------------
# Avaliação
# -----------------------------
def avaliar(sistema: Dict[str, Any], gt: List[Dict[str, Any]], k_list: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]: # sistema: o dicionário retornado contém modelo + índices + docs
    
    #o que a função retorna abaixo

    """
    Retorna:
      - recall_medio[k]
    """
    recall_soma = {k: 0.0 for k in k_list}  # Cria dois dicionários para somar as métricas ao longo das queries:
    #mrr_soma = {k: 0.0 for k in k_list}
    n_validas = 0
    
    
    for item in gt: # itera por cada item do GT, cada item é um dicionário do JSON (um caso de teste).
        query = str(item.get("query", "")).strip() # pega o query se nao te ele  vira ""  /// trip remove espaços do início e fim.
        relevantes = [str(x).strip() for x in item.get("relevantes", []) if str(x).strip()] #  pega a lista de relevantes; se faltar, usa lista vazia.

        cultura = item.get("cultura", None)  # tenta ler a cultura se tiver o retivel vai usar

        if query == "" or not relevantes:  # Se a query ficou vazia ou não tem relevantes pula
            continue

        k_max = max(k_list)  # Pega o maior k da lista.
        resultados = buscar_top_k(sistema, query=query, k=k_max, cultura=cultura) # Executa o retrieval e pede top-k_max de uma vez.

        # (IDs recuperados em ordem de rank)
        recuperados = [] 
        for r in resultados:
            idx = r.get("index", None) # le r["index"] (o ID externo do documento, vindo do BULK)
            if idx is not None:
                recuperados.append(str(idx))

        # DEBUG (temporário)
        print("\nQUERY:", query)
        print("RELEVANTES (GT):", relevantes)
        print(f"TOP-{k_max} (index retornado):", recuperados[:k_max])

        if not (set(relevantes) & set(recuperados)): #calcula interseção.
            print(">> AVISO: nenhum relevante apareceu no top-10.")

        # Acumula métricas (SEMPRE acumula, independente do aviso)
        for k in k_list:
            recall_soma[k] += recall_at_k(relevantes, recuperados, k) # calcula recall_at_k(...) e soma no acumulador
           # mrr_soma[k] += mrr_at_k(relevantes, recuperados, k) # mesma coisa aqui

        n_validas += 1

    if n_validas == 0:
        raise RuntimeError("Nenhuma entrada válida no ground truth (query vazia ou relevantes vazios).")

    recall_medio = {k: recall_soma[k] / n_validas for k in k_list} # Divide cada soma pelo número de queries válidas.
    #mrr_medio = {k: mrr_soma[k] / n_validas for k in k_list}
    return recall_medio #mrr_medio

 
#def imprimir_resultados(recall_medio: Dict[int, float], mrr_medio: Dict[int, float]) -> None: # Recebe os dois dicionários de médias e imprime.
# def imprimir_resultados(recall_medio: Dict[int, float], mrr_medio: Dict[int, float]) -> None:
def imprimir_resultados(recall_medio: Dict[int, float]) -> None:
    print("\n=== Avaliação do Retriever ===")
    print("Métricas médias no conjunto de testes (quanto maior, melhor)\n")

    # Tabela simples (sem pandas)
    print(f"{'k':>3} | {'Recall@k':>10}")
    print("-" * 30)
    for k in sorted(recall_medio.keys()):
        print(f"{k:>3} | {recall_medio[k]:>10.4f}")
    print()


if __name__ == "__main__":
    # 1) Inicializa o sistema (carrega modelo + cria índices FAISS)
    sistema = inicializar_sistema()

    # 2) Carrega ground truth
    gt = carregar_ground_truth(GROUND_TRUTH_PATH)

    # 3) Avalia
    recall_medio = avaliar(sistema, gt, K_LIST)

    # 4) Mostra resultados
    imprimir_resultados(recall_medio)
