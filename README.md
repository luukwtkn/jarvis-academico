# 🤖 JARVIS Acadêmico

> Assistente Pessoal Inteligente para Estudantes
> Trabalho Prático de Inteligência Artificial – **Entrega 2 (Final)**

---

## Funcionalidades implementadas

### Trabalho 1
| # | Funcionalidade | Status |
|---|---|---|
| 3.1 | Consulta a materiais de estudo (RAG) | ✅ |
| 3.2 | Agenda acadêmica (CRUD completo) | ✅ |
| 3.3 | Lista de tarefas (CRUD completo) | ✅ |

### Trabalho 2
| # | Funcionalidade | Status |
|---|---|---|
| 3.4 | Planejamento de estudos | ✅ |
| — | Melhoria de aprendizado 1: Geração de exercícios | ✅ |
| — | Melhoria de aprendizado 2: Active Recall interativo | ✅ |
| — | Avaliação e análise de erros | ✅ |

---

## Arquitetura

```
jarvis/
├── main.py                 ← ponto de entrada (CLI)
├── requirements.txt
├── .env.example             ← copie para .env e configure
├── docs/                    ← coloque seus PDFs e TXTs aqui
├── data/                     ← gerado automaticamente
│   ├── agenda.json
│   ├── tasks.json
│   ├── faiss_index.faiss
│   └── faiss_index_meta.pkl
└── src/
    ├── rag.py                ← 3.1 RAG (chunking, embeddings, FAISS)
    ├── agenda.py              ← 3.2 Agenda (CRUD completo)
    ├── tasks.py                ← 3.3 Tarefas (CRUD completo)
    ├── planejamento.py          ← 3.4 Planejamento de estudos
    ├── aprendizado.py            ← Exercícios + Active Recall
    └── llm.py                     ← Roteamento de intenção + integração LLM
```

---

## Instalação

```bash
# 1. Clone o repositório / descompacte o projeto
cd jarvis

# 2. Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com seu editor preferido e preencha:
#   GEMMA_API_KEY  → token fornecido pelo professor
#   GEMMA_BASE_URL → URL base da API
#   GEMMA_MODEL    → nome do modelo (ex: gemma-12b)
```

---

## Uso

### Setup inicial (primeira vez)

```bash
# Coloque seus PDFs/TXTs na pasta docs/
cp ~/meus_pdfs/*.pdf docs/

# Constrói índice RAG e popula agenda de exemplo
python main.py --rebuild
```

### Iniciando o chat

```bash
python main.py
```

## Guia de uso por funcionalidade

### 3.1 — Consulta a materiais (RAG)
```
Explique regressão logística
Resuma o conteúdo sobre embeddings
Quais são os principais pontos do material de redes neurais?
```

### 3.2 — Agenda acadêmica
```
adicione evento "Prova de BD" amanhã às 17h
edite o evento #2, novo título "Prova de Cálculo II"
remova o evento #3
o que tenho hoje?
quais são minhas aulas esta semana?
```

### 3.3 — Lista de tarefas
```
adicione tarefa "estudar para prova" para 25/06 prioridade alta
edite a tarefa #1, novo prazo 30/06
remova a tarefa #4
marca tarefa 2 como concluída
lista minhas tarefas pendentes
```

### 3.4 — Planejamento de estudos
Combina **agenda + tarefas + materiais (RAG)** num único plano:
```
Monte um plano de estudos para a prova
O que devo priorizar hoje?
Como me preparar para a semana?
```
O sistema identifica provas/entregas próximas na agenda, cruza com tarefas pendentes, busca conteúdo relevante nos materiais e gera um plano com ordem de prioridade e tempo estimado.

### Melhoria 1 — Geração de exercícios
Gera exercícios com gabarito a partir dos PDFs indexados:
```
Gere 3 exercícios sobre árvore de decisão
Me dê questões sobre embeddings
```

### Melhoria 2 — Active Recall (interativo, com avaliação)
O sistema pergunta, o usuário responde, e o sistema avalia e já faz a próxima pergunta — ciclo contínuo até o usuário encerrar.
```
Você: me faça perguntas sobre árvore de decisão
JARVIS: [pergunta 1] ... Sua resposta:

Você: [sua resposta]
JARVIS: [avaliação da resposta 1] + [pergunta 2] ... Sua resposta:

Você: encerrar
JARVIS: [análise das dificuldades identificadas na sessão]
```

### Avaliação e análise de erros
Ao encerrar uma sessão de Active Recall, o sistema analisa o histórico de respostas do usuário e identifica:
- Temas com mais dificuldade
- Tópicos prioritários para revisão
- Estratégia de estudo personalizada

Pode também ser chamado isoladamente após uma sessão:
```
Identifique minhas dificuldades
```

---

## Como funciona — Fluxo geral

```
Usuário
  │
  ▼
main.py (CLI)
  │ mensagem + histórico
  ▼
llm.py — detecção de intenção (palavras-chave)
  │
  ├─ agenda_add/edit/del/consulta ──▶ agenda.py  (data/agenda.json)
  ├─ tarefas_add/edit/del/done/list ▶ tasks.py   (data/tasks.json)
  ├─ rag ───────────────────────────▶ rag.py     (FAISS + embeddings)
  ├─ planejamento ──────────────────▶ planejamento.py (agenda + tasks + rag)
  ├─ exercicios ────────────────────▶ aprendizado.py (RAG + prompt)
  ├─ recall_iniciar/resposta ───────▶ aprendizado.py (estado de sessão)
  └─ dificuldades ───────────────────▶ aprendizado.py (histórico da sessão)
              │
              ▼
     contexto montado é injetado no prompt
              │
              ▼
        LLM gera resposta final
              │
              ▼
          Usuário
```

---

## Detalhes técnicos

### RAG (3.1)
- **Leitura:** PyMuPDF (PDF) + leitura direta (TXT, MD)
- **Chunking:** janela deslizante de 500 caracteres, 100 de sobreposição
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Índice vetorial:** FAISS `IndexFlatL2`, persistido em disco
- **Recuperação:** top-4 chunks por distância L2

### Agenda (3.2)
- Armazenada em `data/agenda.json`
- Tipos: aula, prova, entrega, reunião, outro
- Filtros por período: hoje, amanhã, semana, todos
- Filtros por tipo de evento

### Tarefas (3.3)
- Armazenadas em `data/tasks.json`
- Prioridades: alta 🔴, média 🟡, baixa 🟢
- Campos: título, disciplina, prazo, prioridade, status

### Planejamento (3.4)
- `gerar_contexto_planejamento()` combina:
  - Eventos da semana (`agenda.eventos_semana()`)
  - Provas e entregas com contagem de dias restantes
  - Tarefas pendentes ordenadas por prioridade
  - Trechos relevantes dos materiais (se um tema for mencionado)

### Active Recall (Melhoria de aprendizado)
- Estado de sessão mantido em memória (`_active_recall_state`)
- Cada resposta do usuário gera, em uma única chamada ao LLM:
  1. Avaliação da resposta anterior (Correto / Parcial / Incorreto + feedback)
  2. Nova pergunta sobre o mesmo tema
- Ao encerrar (`"encerrar"`, `"parar"`, `"sair"`), o histórico da sessão é analisado para identificar dificuldades

### IAs Utilizadas:
- Claude.ai (Sonnet 4.6)
- ChatGPT (GPT-5.5)
- Gemini (3.5 Thinking)
---

## Dependências principais

| Biblioteca | Uso |
|---|---|
| openai | Cliente para API do LLM (compatível com Gemma/Qwen) |
| sentence-transformers | Embeddings locais |
| faiss-cpu | Busca vetorial |
| PyMuPDF | Leitura de PDF |
| rich | Interface CLI |
| python-dotenv | Variáveis de ambiente |


## Autores

- André Augusto Silveira

Disciplina: Inteligência Artificial  
Professor: Edson Takashi Matsubara
Semestre: 2026/1
