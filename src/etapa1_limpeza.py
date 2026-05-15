"""
src/etapa1_limpeza.py

Pipeline de limpeza para o arquivo dados/noticias_brutas.json.
Remove tags HTML, decodifica entidades, normaliza Unicode, remove timestamps/metadados embutidos e colapsa whitespace.

Entrada: dados/noticias_brutas.json
Saída:   dados/noticias_limpas.json

Executar a partir da raiz do projeto:
    python -m src.etapa1_limpeza
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

ENTRADA = Path("dados/noticias_brutas.json")
SAIDA = Path("dados/noticias_limpas.json")

# Regex de metadados embutidos no corpo do texto.
PADROES_METADADOS = [
    re.compile(
        r"Publicado em:\s*\d{1,2}/\d{1,2}/\d{4}(?:\s*às?\s*\d{1,2}h\d{0,2})?",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}\s*-\s*\d{1,2}h\d{0,2}\s*\|\s*\w+"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}\s*[—–-]\s*[^\n.]{1,40}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}"),
]


def limpar_texto(texto_bruto: str) -> str:
    # Remove tags HTML; BS4 também decodifica entidades.
    texto = BeautifulSoup(texto_bruto, "lxml").get_text(separator=" ")
    # Salvaguarda contra entidades remanescentes.
    texto = html.unescape(texto)
    # Normaliza forma Unicode.
    texto = unicodedata.normalize("NFKC", texto)

    # Remove metadados embutidos (timestamps, fontes no header).
    for padrao in PADROES_METADADOS:
        texto = padrao.sub(" ", texto)

    # Remove espaços antes de pontuação (efeito do separador do BS4).
    texto = re.sub(r"\s+([.,;:!?])", r"\1", texto)
    # Colapsa quebras de linha e espaços múltiplos.
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def carregar_noticias(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return json.load(f)


def salvar_noticias(noticias: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        json.dump(noticias, f, ensure_ascii=False, indent=2)


def main() -> None:
    noticias = carregar_noticias(ENTRADA)
    print(f"[etapa1] {len(noticias)} notícias carregadas de {ENTRADA}")

    limpas: list[dict] = []
    for n in noticias:
        texto_limpo = limpar_texto(n["texto"])
        limpas.append(
            {
                "id": n["id"],
                "titulo": n["titulo"].strip(),
                "texto": texto_limpo,
                "data": n["data"],
                "fonte": n["fonte"],
                "n_caracteres": len(texto_limpo),
            }
        )

    salvar_noticias(limpas, SAIDA)
    print(f"[etapa1] {len(limpas)} notícias limpas salvas em {SAIDA}")

    # Sinaliza casos extremos (texto muito curto após limpeza).
    curtas = [n for n in limpas if n["n_caracteres"] < 30]
    if curtas:
        print(f"[etapa1] Aviso: {len(curtas)} notícia(s) com texto muito curto:")
        for n in curtas:
            print(f"  - id={n['id']}: {n['texto']!r} ({n['n_caracteres']} chars)")


if __name__ == "__main__":
    main()