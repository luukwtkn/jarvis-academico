# 🤖 JARVIS Acadêmico

> Assistente Pessoal Inteligente para Estudantes  
> Trabalho Prático de Inteligência Artificial – Entrega 1

---

## Funcionalidades implementadas

| # | Funcionalidade | Status |
|---|---------------|--------|
| 3.1 | Consulta a materiais de estudo (RAG) | ✅ |
| 3.2 | Agenda acadêmica | ✅ |
| 3.3 | Lista de tarefas | ✅ |

---

## Arquitetura

```
jarvis/
├── main.py               ← ponto de entrada (CLI)
├── requirements.txt
├── .env.example          ← copie para .env e configure
├── docs/                 ← coloque seus PDFs e TXTs aqui
├── data/                 ← gerado automaticamente
│   ├── agenda.json
│   ├── tasks.json
│   ├── faiss_index.faiss
│   └── faiss_index_meta.pkl
└── src/
    ├── rag.py            ← Funcionalidade 3.1 (RAG)
    ├── agenda.py         ← Funcionalidade 3.2 (Agenda)
    ├── tasks.py          ← Funcionalidade 3.3 (Tarefas)
    └── llm.py            ← Integração LLM + Tool Calling
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

### Exemplos de uso

**Consulta RAG (materiais de estudo):**
```
Você: Explique o que é regressão logística
Você: Resuma o conteúdo sobre embeddings
Você: Quais são os principais pontos do material de redes neurais?
```

**Agenda:**
```
Você: O que tenho hoje?
Você: Quais são minhas aulas esta semana?
Você: Tenho prova amanhã?
```

**Tarefas:**
```
Você: Adiciona tarefa: estudar para prova de cálculo
Você: Lista minhas tarefas pendentes
Você: Marca tarefa 2 como concluída
```

---

## Como funciona – Diagrama de fluxo

```
Usuário
  │
  ▼
main.py (CLI)
  │  mensagem + histórico
  ▼
llm.py (Gemma 12B)
  │
  ├─ tool_call: consultar_materiais ──▶ rag.py
  │                                      ├─ embed(query)
  │                                      ├─ FAISS.search()
  │                                      └─ retorna trechos relevantes
  │
  ├─ tool_call: consultar_agenda ─────▶ agenda.py
  │                                      └─ lê agenda.json filtrado
  │
  └─ tool_call: gerenciar_tarefas ────▶ tasks.py
                                         └─ CRUD em tasks.json
                                              │
                                              ▼
                                    Resultado da ferramenta
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
- **Leitura de documentos:** PyMuPDF (PDF) + leitura direta (TXT, MD)
- **Chunking:** janela deslizante com 500 chars e 100 chars de sobreposição
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, sem GPU)
- **Índice vetorial:** FAISS IndexFlatL2 (persistido em disco)
- **Recuperação:** top-4 chunks mais similares por cosine/L2

### Tool Calling
- Implementado no padrão OpenAI function calling
- O LLM decide qual ferramenta usar com base na intenção do usuário
- Suporte a múltiplas ferramentas por turno

### Agenda (3.2)
- Armazenada em `data/agenda.json`
- Tipos: aula, prova, entrega, reunião, outro
- Filtros por período: hoje, amanhã, semana, todos
- Filtros por tipo de evento

### Tarefas (3.3)
- Armazenadas em `data/tasks.json`
- Prioridades: alta 🔴, média 🟡, baixa 🟢
- Campos: título, disciplina, prazo, prioridade, status

### IAs Utilizadas:
- Claude.ai (Sonnet 4.6)
- ChatGPT (GPT-5.5)
- Gemini (3.5 Thinking)
---

## Dependências principais

| Biblioteca | Versão mínima | Uso |
|-----------|-------------|-----|
| openai | 1.0.0 | Cliente para Gemma 12B |
| sentence-transformers | 2.2.0 | Embeddings locais |
| faiss-cpu | 1.7.4 | Busca vetorial |
| PyMuPDF | 1.23.0 | Leitura de PDF |
| rich | 13.0.0 | Interface CLI |
| python-dotenv | 1.0.0 | Variáveis de ambiente |

---

## Autores

- André Augusto Silveira

Disciplina: Inteligência Artificial  
Professor: Edson Takashi Matsubara
Semestre: 2025/1
