"""
jarvis/src/aprendizado.py
=========================
Melhorias de Aprendizado
1. Geração de exercícios baseados nos materiais (RAG)
2. Active Recall interativo — sistema pergunta e avalia respostas
"""


def gerar_prompt_exercicios(contexto_rag: str, tema: str, quantidade: int = 3) -> str:
    """
    Monta o prompt para o LLM gerar exercícios baseados nos materiais.
    """
    return f"""Com base nos trechos dos materiais de estudo abaixo, gere {quantidade} exercícios
sobre o tema "{tema}". Para cada exercício:
1. Faça uma pergunta clara e objetiva
2. Forneça a resposta correta ao final
3. Indique o nível de dificuldade (Fácil / Médio / Difícil)

Formato obrigatório para cada exercício:
---
**Exercício N** [Nível]
Pergunta: ...
Resposta: ...
---

MATERIAIS:
{contexto_rag}

Gere os exercícios agora:"""


def gerar_prompt_active_recall(contexto_rag: str, tema: str) -> str:
    """
    Monta o prompt para o LLM iniciar uma sessão de active recall.
    O LLM faz UMA pergunta e aguarda a resposta do usuário.
    """
    return f"""Você é um tutor de active recall. Com base nos materiais abaixo sobre "{tema}",
faça UMA pergunta desafiadora ao estudante para testar seu conhecimento.

Regras:
- Faça apenas UMA pergunta por vez
- A pergunta deve exigir raciocínio, não só memorização
- Não forneça a resposta ainda — aguarde o estudante responder
- Termine com: "Sua resposta:"

MATERIAIS:
{contexto_rag}

Faça a pergunta agora:"""


def gerar_prompt_avaliar_resposta(pergunta: str, resposta_usuario: str, contexto_rag: str) -> str:
    """
    Monta o prompt para o LLM avaliar a resposta do usuário no active recall.
    """
    return f"""Você é um tutor avaliando a resposta de um estudante.

PERGUNTA FEITA:
{pergunta}

RESPOSTA DO ESTUDANTE:
{resposta_usuario}

MATERIAIS DE REFERÊNCIA:
{contexto_rag}

Avalie a resposta seguindo este formato:
✅ **Correto** / ⚠️ **Parcialmente correto** / ❌ **Incorreto**

**Feedback:** (explique o que acertou e o que errou)
**Resposta completa:** (dê a resposta ideal com base nos materiais)
**Dica para fixar:** (sugira uma forma de memorizar melhor)"""


def gerar_prompt_identificar_dificuldades(historico_perguntas: list) -> str:
    """
    Analisa o histórico de perguntas do usuário para identificar dificuldades.
    """
    perguntas_str = "\n".join(f"- {p}" for p in historico_perguntas)
    return f"""Analise as perguntas feitas por este estudante durante a sessão de estudos:

{perguntas_str}

Com base nisso:
1. Identifique os temas onde o estudante demonstra mais dificuldade
2. Sugira tópicos para revisar com prioridade
3. Recomende uma estratégia de estudo personalizada

Seja específico e construtivo."""
