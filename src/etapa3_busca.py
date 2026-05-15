"""
src/etapa3_busca.py

Motor de busca semântica sobre o corpus de notícias. Carrega o modelo intfloat/multilingual-e5-small 
e a matriz de embeddings pré-computada, recebe uma query em texto livre e retorna os top-K artigos mais similares.

Decisões:
-Prefixo "query: " conforme convenção da família E5 (documentos usam
  "passage: " e foi aplicado na Etapa 2).
-Embeddings já L2-normalizados na Etapa 2: cosine similarity vira
  produto interno simples (embeddings @ query_emb).
-Modelo e embeddings são carregados uma única vez no main.

Modos de uso (a partir da raiz do projeto):
    (roda as 3 queries do desafio)
    python -m src.etapa3_busca

    (busca livre)
    python -m src.etapa3_busca --query "sua consulta aqui" --k 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

NOTICIAS_PATH = Path("dados/noticias_limpas.json")
EMBEDDINGS_PATH = Path("dados/embeddings.npy")

NOME_MODELO = "intfloat/multilingual-e5-small"
PREFIXO_QUERY = "query: "

QUERIES_DESAFIO = [
    "mudanças na taxa de juros",
    "mercado de trabalho e desemprego",
    "inflação e preços ao consumidor",
]


@dataclass
class Resultado:
    # Linha de saída da busca semântica.

    id: int
    titulo: str
    fonte: str
    data: str
    score: float
    snippet: str


def carregar_corpus() -> tuple[list[dict], np.ndarray]:
    # Carrega notícias e embeddings alinhados por índice.
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} não encontrado. "
            "Rode antes: python -m src.etapa2_embeddings"
        )

    with NOTICIAS_PATH.open(encoding="utf-8") as f:
        noticias = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(noticias) != embeddings.shape[0]:
        raise ValueError(
            f"Desalinhamento: {len(noticias)} notícias vs "
            f"{embeddings.shape[0]} embeddings. "
            "Reexecute a Etapa 2 para regenerar."
        )

    return noticias, embeddings


def buscar(
    query: str,
    noticias: list[dict],
    embeddings: np.ndarray,
    modelo: SentenceTransformer,
    k: int = 3,
) -> list[Resultado]:
    # Retorna top-K resultados ordenados por similaridade decrescente.
    if not query.strip():
        raise ValueError("Query vazia.")

    query_emb = modelo.encode(
        f"{PREFIXO_QUERY}{query}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Embeddings já normalizados: cosine similarity = dot product.
    scores = embeddings @ query_emb

    # argsort decrescente, top-K.
    top_indices = np.argsort(-scores)[:k]

    resultados = []
    for idx in top_indices:
        noticia = noticias[int(idx)]
        snippet = noticia["texto"][:200]
        if len(noticia["texto"]) > 200:
            snippet += "..."
        resultados.append(
            Resultado(
                id=noticia["id"],
                titulo=noticia["titulo"],
                fonte=noticia["fonte"],
                data=noticia["data"],
                score=float(scores[idx]),
                snippet=snippet,
            )
        )
    return resultados


def imprimir_resultados(query: str, resultados: list[Resultado]) -> None:
    # Formata e imprime resultados no terminal.
    print(f'\nQuery: "{query}"')
    print(f"Top {len(resultados)} resultados:\n")
    for i, r in enumerate(resultados, start=1):
        print(f"  #{i} [score={r.score:.4f}] id={r.id} — {r.fonte} ({r.data})")
        print(f"      Título: {r.titulo}")
        print(f"      Trecho: {r.snippet}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motor de busca semântica sobre o corpus de notícias."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Consulta em texto livre. Se omitido, executa as 3 queries do desafio.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Número de resultados a retornar (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    noticias, embeddings = carregar_corpus()
    print(f"[etapa3] Corpus: {len(noticias)} notícias, embeddings {embeddings.shape}")
    print(f"[etapa3] Carregando modelo {NOME_MODELO}...")
    modelo = SentenceTransformer(NOME_MODELO)
    print("[etapa3] Modelo carregado.")

    queries = [args.query] if args.query else QUERIES_DESAFIO

    for q in queries:
        resultados = buscar(q, noticias, embeddings, modelo, k=args.k)
        imprimir_resultados(q, resultados)


if __name__ == "__main__":
    main()