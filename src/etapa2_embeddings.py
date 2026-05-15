"""
src/etapa2_embeddings.py

Gera embeddings densos para cada notícia do arquivo limpo, usando
o modelo intfloat/multilingual-e5-small.

Decisões:
-Modelo E5 multilingual: treinado especificamente para retrieval,
  com suporte robusto a português via contrastive learning.
-Prefixo "passage: " conforme convenção da família E5 para documentos
  do corpus (queries usam "query: ", aplicado na Etapa 3).
-Concatenação "titulo. texto" antes do embedding para enriquecer o
  vetor com o sinal denso do título.
-Vetores L2-normalizados: similaridade cosseno vira dot product simples.

Entrada: dados/noticias_limpas.json
Saída:   dados/embeddings.npy  (matriz NxD, alinhada por índice com noticias_limpas.json)

Executar a partir da raiz do projeto:
    python -m src.etapa2_embeddings
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ENTRADA = Path("dados/noticias_limpas.json")
SAIDA = Path("dados/embeddings.npy")

NOME_MODELO = "intfloat/multilingual-e5-small"
PREFIXO_DOCUMENTO = "passage: "


def carregar_noticias(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return json.load(f)


def formatar_documento(noticia: dict) -> str:
    # Concatena título e texto com prefixo passage para E5.
    titulo = noticia["titulo"].strip()
    texto = noticia["texto"].strip()
    return f"{PREFIXO_DOCUMENTO}{titulo}. {texto}"


def gerar_embeddings(
    textos: list[str], modelo: SentenceTransformer
) -> np.ndarray:
    # Encoda lista de textos em matriz NxD normalizada.
    return modelo.encode(
        textos,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def main() -> None:
    noticias = carregar_noticias(ENTRADA)
    print(f"[etapa2] {len(noticias)} notícias carregadas de {ENTRADA}")

    print(f"[etapa2] Carregando modelo {NOME_MODELO}...")
    t0 = time.perf_counter()
    modelo = SentenceTransformer(NOME_MODELO)
    print(f"[etapa2] Modelo carregado em {time.perf_counter() - t0:.1f}s")

    documentos = [formatar_documento(n) for n in noticias]

    print(f"[etapa2] Gerando embeddings de {len(documentos)} documentos...")
    t0 = time.perf_counter()
    embeddings = gerar_embeddings(documentos, modelo)
    print(f"[etapa2] Embeddings gerados em {time.perf_counter() - t0:.1f}s")
    print(f"[etapa2] Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    np.save(SAIDA, embeddings)
    print(f"[etapa2] Salvo em {SAIDA} ({SAIDA.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()