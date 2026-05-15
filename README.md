# Desafio Técnico — Estágio em Ciência de Dados (FGV IBRE)

Mini motor de busca semântico aplicado a notícias econômicas.

O pipeline está dividido em três etapas encadeadas: limpeza de texto, geração de embeddings densos e busca por similaridade semântica. Cada etapa é um script Python autocontido, executável de forma independente.

```
noticias_brutas.json
        ↓ (Etapa 1 — limpeza)
noticias_limpas.json
        ↓ (Etapa 2 — embeddings)
embeddings.npy
        ↓ (Etapa 3 — busca)
top-K resultados ranqueados
```

---

## Estrutura do projeto

```
desafio-dados-ibre/
    dados/
        noticias_brutas.json    # entrada original (20 notícias)
        noticias_limpas.json    # saída da Etapa 1
        embeddings.npy          # gerado pela Etapa 2 (não versionado)
    src/
        __init__.py
        etapa1_limpeza.py
        etapa2_embeddings.py
        etapa3_busca.py
    .gitignore
    LICENSE
    README.md
    requirements.txt
```

---

## Como rodar

### Pré-requisitos

- Python 3.14 (testado nessa versão; deve funcionar em 3.10+)
- Git
- ~500 MB de espaço em disco (modelo de embedding é baixado na primeira execução)

### Setup

```bash
# 1) clonar o repositório
git clone https://github.com/YuriFMX/desafio-dados-ibre.git
cd desafio-dados-ibre

# 2) criar e ativar ambiente virtual
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3) instalar dependências
pip install -r requirements.txt
```

### Executar o pipeline completo

```bash
python -m src.etapa1_limpeza      # gera dados/noticias_limpas.json
python -m src.etapa2_embeddings   # gera dados/embeddings.npy
python -m src.etapa3_busca        # roda as 3 queries do desafio
```

### Busca livre

```bash
python -m src.etapa3_busca --query "sua consulta aqui" --k 5
```

Argumentos:
- `--query`: texto livre da consulta. Se omitido, executa as 3 queries do desafio.
- `--k`: número de resultados a retornar (default: 3).

---

## Decisões técnicas

### Etapa 1 — Limpeza e tratamento de texto

**Pipeline aplicado a cada texto bruto** (em ordem):

1. **Remoção de tags HTML** com `BeautifulSoup` + parser `lxml`. O método `.get_text(separator=" ")` já decodifica entidades HTML simultaneamente.
2. **Decodificação adicional** com `html.unescape`** para entidades remanescentes (`&agrave;`, `&ccedil;`, etc.).
3. **Normalização Unicode** com `unicodedata.normalize("NFKC", ...)` para garantir forma canônica dos caracteres acentuados.
4. **Remoção de metadados embutidos** (timestamps de publicação, fontes no cabeçalho) via 4 regex aplicadas em ordem do mais específico para o mais genérico:
   - `Publicado em: DD/MM/YYYY [às HHhMM]`
   - `DD/MM/YYYY - HHhMM | Categoria`
   - `DD/MM/YYYY — Fonte/nota curta`
   - Data nua `DD/MM/YYYY`
5. **Remoção de espaços antes de pontuação** — efeito colateral do separador do BeautifulSoup, que insere espaços em locais onde havia tags inline (ex.: `pelo <strong>IBGE</strong>.` → `pelo IBGE .`).
6. **Colapso final de whitespace** (`\s+` → ` `) e `strip()`.

**Decisões deliberadas de *não* fazer**:

- **Não fazer lowercasing**: modelos de embedding modernos são case-sensitive e preservam morfologia. Lowercasing perde sinal (ex.: "FGV IBRE" vs "fgv ibre").
- **Não remover acentos**: pelas mesmas razões; tokenizadores modernos lidam bem com acentos.
- **Não remover stopwords**: stopwords contribuem para o contexto semântico em modelos de embedding.

**Tratamento do caso extremo (id 18)**: a notícia "Nota curta" tem como texto apenas `"   Selic."`. Após limpeza, fica `"Selic."` (6 caracteres). O caso é preservado deliberadamente — filtrar antes de embeddar esconderia o problema. O script emite um aviso no terminal informando textos abaixo de 30 caracteres.

### Etapa 2 — Geração de embeddings

**Modelo escolhido**: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)

**Justificativa**:

| Critério | Análise |
|---|---|
| Idioma do corpus | Português brasileiro, modelos apenas em inglês ficam fora. |
| Tarefa | Retrieval assimétrico (query curta → documento longo). |
| Adequação ao treino | A família E5 foi treinada com contrastive learning especificamente para retrieval, com a convenção de prefixos `query:` para consultas e `passage:` para documentos. |
| Tamanho | ~118 MB (variante `small`), roda em qualquer máquina sem GPU em milissegundos. |
| Cobertura | Treinado em 100+ idiomas, com cobertura sólida de português. |

**Convenções aplicadas**:

- **Prefixo `passage: `** prepended a cada documento, conforme convenção da família E5.
- **Concatenação `titulo. texto`** antes do embedding — títulos carregam sinal semântico denso e enriquecem o vetor sem custo adicional.
- **Normalização L2 dos vetores** (`normalize_embeddings=True`). Com vetores normalizados, similaridade cosseno vira produto interno simples — mais rápido sem perda de precisão.

**Saída**: matriz NumPy de shape `(20, 384)`, dtype `float32`, salva em `dados/embeddings.npy` (~30 KB).

### Etapa 3 — Motor de busca semântica

**Algoritmo**:

1. Recebe query em texto livre.
2. Aplica prefixo `query: ` (convenção E5).
3. Encoda a query com o mesmo modelo da Etapa 2, com `normalize_embeddings=True`.
4. Calcula similaridade via produto interno entre o vetor da query e a matriz de embeddings dos documentos: `scores = embeddings @ query_emb`.
5. Retorna top-K documentos ordenados por score decrescente.

**Decisões de implementação**:

- Modelo e embeddings são carregados **uma única vez** no `main()`. Releitura a cada query seria proibitivo dado o tempo de inicialização do modelo (~16s).
- O alinhamento entre `noticias_limpas.json` e `embeddings.npy` é feito por índice posicional (linha N do JSON corresponde à linha N da matriz). Uma verificação de consistência é executada no carregamento.
- Interface por linha de comando com `argparse`: dois modos (queries do desafio ou query livre) com escolha de K.

---

## Resultados qualitativos

Resultados obtidos rodando `python -m src.etapa3_busca` na ordem em que aparecem no terminal.

### Query 1: "mudanças na taxa de juros"

| Rank | Score | id | Fonte / Data | Título | Análise |
|---|---|---|---|---|---|
| 1 | 0.8658 | 1 | Banco Central / 2023-08-02 | Copom mantém Selic em 13,75% ao ano pela quarta reunião consecutiva | **Relevante** — trata diretamente de decisão sobre Selic. |
| 2 | 0.8614 | 7 | Banco Central / 2023-08-25 | Inadimplência das famílias sobe para 6,3% em julho, aponta BC | **Tangencial** — menciona "efeito defasado dos juros altos" mas o foco é inadimplência. |
| 3 | 0.8547 | 15 | Reuters Brasil / 2023-08-28 | Câmbio: real se fortalece com melhora do ambiente externo e fiscal | **Tangencial** — menciona "aperto monetário nos EUA" no contexto cambial. |

**Observação**: as duas notícias mais diretamente relevantes para a query, sendo elas o id 6 ("Copom inicia ciclo de corte e reduz Selic para 13,25%") e id 11 ("Selic deve recuar a 9% até o fim de 2024"), ficaram fora do top-3, apesar de serem os exemplos mais explícitos de mudança na taxa de juros do corpus. Discussão na seção de Limitações.

### Query 2: "mercado de trabalho e desemprego"

| Rank | Score | id | Fonte / Data | Título | Análise |
|---|---|---|---|---|---|
| 1 | 0.8829 | 4 | IBGE / 2023-08-29 | Taxa de desemprego cai para 7,9% no segundo trimestre, menor nível desde 2014 | **Relevante** — match direto. |
| 2 | 0.8576 | 14 | IBGE / 2023-08-30 | Desemprego juvenil no Brasil ainda preocupa apesar de melhora geral | **Relevante** — recorte específico do mesmo tema. |
| 3 | 0.8429 | 16 | FGV IBRE / 2023-08-30 | IGP-M registra terceira deflação consecutiva em agosto | **Falso positivo** — texto trata de índice de preços, sem menção a trabalho ou desemprego. |

**Observação**: top-2 é exatamente o esperado. A terceira posição poderia ter sido preenchido por id 17 (ICC menciona "queda do desemprego") ou id 19 (PMS menciona "mercado de trabalho").

### Query 3: "inflação e preços ao consumidor"

| Rank | Score | id | Fonte / Data | Título | Análise |
|---|---|---|---|---|---|
| 1 | 0.8909 | 9 | FGV IBRE / 2023-08-10 | Inflação ao produtor (IPA) desacelera e pressão sobre preços finais diminui | **Relevante** — IPA é índice de preços diretamente conectado. |
| 2 | 0.8860 | 16 | FGV IBRE / 2023-08-30 | IGP-M registra terceira deflação consecutiva em agosto | **Relevante** — IGP-M agrega IPA, IPC e INCC. |
| 3 | 0.8829 | 2 | IBGE / 2023-08-09 | IPCA de julho registra 0,12%, menor resultado para o mês desde 2006 | **Relevante** — IPCA é literalmente o "Índice Nacional de Preços ao Consumidor Amplo". |

**Observação**: os três resultados pertencem todos à família correta de indicadores. Os scores estão muito próximos entre si (delta < 0.01), o que sugere baixa diferenciação dentro dessa família temática.

### Query bônus: "expectativas do mercado financeiro" (k=5)

| Rank | Score | id | Fonte / Data | Título |
|---|---|---|---|---|
| 1 | 0.8828 | 20 | Valor Econômico / 2023-09-05 | Expectativas para o PIB de 2023 sobem para 2,5% após resultado do segundo trimestre |
| 2 | 0.8666 | 11 | Banco Central / 2023-08-15 | Selic deve recuar a 9% até o fim de 2024, projetam economistas |
| 3 | 0.8583 | 15 | Reuters Brasil / 2023-08-28 | Câmbio: real se fortalece com melhora do ambiente externo e fiscal |
| 4 | 0.8570 | 17 | FGV IBRE / 2023-08-29 | Confiança do consumidor sobe pelo quarto mês seguido em agosto |
| 5 | 0.8546 | 10 | Banco Central / 2023-08-25 | Crédito total no Brasil atinge R$ 5,6 trilhões com desaceleração no crescimento |

**Observação**: top-2 muito coerente, onde o id 20 traz literalmente "expectativas" no título, e id 11 é justamente o Relatório Focus (publicação canônica de expectativas do mercado financeiro). Posições 3-5 são razoáveis.

### Sobre o caso extremo (id 18)

A notícia com texto mínimo ("Selic.", 6 caracteres após limpeza) não apareceu em nenhuma das queries acima, nem mesmo na primeira, em que "Selic" é palavra-chave. O embedding gerado para um texto tão curto fica suficientemente "fraco" para não pontuar alto em nenhuma consulta. O pipeline lidou naturalmente com o caso extremo sem necessidade de filtros específicos.

---

## Limitações observadas

**1. Captura de conceitos implícitos é limitada**

Na Query 1 ("mudanças na taxa de juros"), o conceito-chave é a noção de mudança. As notícias que descrevem mudanças concretas (id 6 — corte; id 11 — projeção de queda) ficaram fora do top-3, enquanto notícias que apenas mencionam "taxa", "Selic" ou "juros" em contextos tangenciais (id 7, id 15) foram ranqueadas acima.

**Hipótese**: o E5 dá peso considerável a entidades nomeadas e tópicos centrais ("Selic", "Banco Central"), com menos sensibilidade a verbos de ação ou a relações conceituais sutis como 'mudança', 'redução', e 'projeção'.

**2. Falsos positivos por proximidade temática genérica**

Na Query 2, o terceiro resultado foi a notícia do IGP-M (deflação), que não menciona trabalho nem desemprego. O modelo provavelmente puxou pela proximidade geral entre "indicadores econômicos do mês", uma similaridade de 'tópico' que não é a mesma coisa que similaridade de 'conteúdo'.

**3. Scores absolutos pouco diferenciados**

Os scores ficaram concentrados na faixa 0.84–0.89, com diferenças pequenas (~0.01–0.03) entre o top-1 e o top-3. Isso indica que o ranking pode ser frágil para pequenas perturbações no corpus ou na query. Em produção, valeria definir um threshold absoluto de relevância ou um delta mínimo entre o melhor e os demais.

**4. Corpus muito pequeno**

Com apenas 20 documentos, qualquer análise quantitativa fica estatisticamente fraca. As observações qualitativas acima são consistentes mas não são generalizáveis para um corpus de produção.

---

## Próximos passos para investigação

Mantendo o escopo proporcional ao desafio, deixei de fora otimizações que aumentariam a complexidade sem benefício claro para 20 documentos. Em um cenário real:

- **Reranking com cross-encoder** (ex.: BAAI/bge-reranker-base) aplicado aos top-K candidatos do retrieval inicial. Cross-encoders capturam interações query-documento melhor que bi-encoders, ao custo de não escalar para corpus grandes, o que é exatamente o caso de uso ideal de reranking sobre top-K.
- **Hybrid search**: combinar busca densa (embeddings) com busca esparsa (BM25), de forma que palavras-chave de ação ("corte", "redução", "alta") tenham peso explícito além da similaridade semântica.
- **Avaliação quantitativa**: construir um conjunto pequeno de queries com julgamento de relevância (mesmo binário: relevante/não-relevante) para calcular métricas padronizadas. Permite comparar variantes (modelos, prefixos, estratégias de concatenação) de forma objetiva.
- **Chunking** para documentos longos. Não aplicável a este corpus, em que cada notícia cabe confortavelmente dentro do limite de tokens do E5, mas é o padrão da indústria para documentos extensos.
- **Indexação vetorial** (FAISS, HNSW) para corpus em escala, sendo desnecessária para 20 documentos, em que busca exaustiva via produto interno custa microssegundos.

---

## Stack técnica

- **Python 3.14.5**
- **sentence-transformers** (`intfloat/multilingual-e5-small`) — geração de embeddings
- **BeautifulSoup4 + lxml** — parsing de HTML
- **NumPy** — operações vetoriais
- **scikit-learn** — listada em 'requirements.txt' mas usada apenas como método de segurança; o cálculo de similaridade é feito com NumPy puro

---

## Reprodutibilidade

Todas as dependências estão fixadas em versões exatas em 'requirements.txt'. Os artefatos derivados ('dados/embeddings.npy') não são versionados — são regenerados pela Etapa 2 em ~20 segundos no primeiro uso (incluindo download do modelo) e ~1 segundo em execuções subsequentes (modelo em cache local).

Para validar a reprodutibilidade do zero:

```bash
git clone https://github.com/YuriFMX/desafio-dados-ibre.git
cd desafio-dados-ibre
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # ou: source .venv/bin/activate
pip install -r requirements.txt
python -m src.etapa1_limpeza
python -m src.etapa2_embeddings
python -m src.etapa3_busca
```
