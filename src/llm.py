"""
jarvis/src/llm.py
=================
Cliente LLM – integração com Gemma/Qwen via API compatível OpenAI
Base v6 + CRUD agenda/tarefas + Funcionalidade 3.4 + Melhorias de aprendizado
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

from src.rag         import RAGSystem
from src             import agenda as agenda_mod
from src             import tasks  as tasks_mod
from src.planejamento import gerar_contexto_planejamento
from src.aprendizado  import (
    gerar_prompt_exercicios,
    gerar_prompt_active_recall,
    gerar_prompt_avaliar_resposta,
    gerar_prompt_identificar_dificuldades,
)

# Carrega .env pelo caminho absoluto
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

client = OpenAI(
    api_key  = os.getenv("GEMMA_API_KEY", ""),
    base_url = os.getenv("GEMMA_BASE_URL", "https://api.example.com/v1"),
)
MODEL = os.getenv("GEMMA_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")

rag = RAGSystem()

SYSTEM_PROMPT = """Você é o JARVIS Acadêmico, um assistente pessoal inteligente para estudantes universitários.

Quando houver informações de contexto fornecidas (dentro de blocos ===), use APENAS essas informações.
Não invente eventos, tarefas ou conteúdos que não estejam no contexto.
Se uma ação foi realizada (adicionar, editar, remover), confirme de forma clara ao usuário.
Responda sempre em português brasileiro de forma direta e amigável.
"""

# ---------------------------------------------------------------------------
# Estado da sessão de active recall
# ---------------------------------------------------------------------------
_active_recall_state: Dict = {
    "ativo":       False,
    "tema":        "",
    "pergunta":    "",
    "contexto":    "",
    "historico":   [],   # perguntas feitas na sessão
}

# ---------------------------------------------------------------------------
# Palavras-chave
# ---------------------------------------------------------------------------

ADD_KW  = ["adicione", "adiciona", "adicionar", "crie", "cria", "criar",
            "insira", "insere", "inserir", "coloque", "coloca", "registre",
            "registra", "nova", "novo", "add", "agende"]

EDIT_KW = ["edite", "edita", "editar", "altere", "altera", "alterar",
            "mude", "muda", "mudar", "atualize", "atualiza", "atualizar",
            "modifique", "modifica", "corrija", "troque", "troca"]

DEL_KW  = ["remova", "remove", "remover", "delete", "deleta", "deletar",
            "exclua", "exclui", "excluir", "apague", "apaga", "apagar",
            "cancele", "cancela", "cancelar"]

DONE_KW = ["concluir", "conclu", "marcar", "marca", "marque",
            "feita", "feito", "done", "finaliz", "completa", "completar",
            "terminei", "terminar"]

AGENDA_KW  = ["evento", "eventos", "agenda", "aula", "aulas", "prova", "provas",
               "compromisso", "entrega", "calendário", "horário", "reunião"]

PERIODO_KW = ["hoje", "amanhã", "amanha", "semana", "o que tenho", "quais"]

TASKS_KW   = ["tarefa", "tarefas", "todo", "pendente", "pendentes",
               "minhas tarefas", "listar tarefas"]

RAG_KW     = ["explique", "explica", "resumo", "resuma", "o que é", "o que são",
               "como funciona", "fale sobre", "material", "conteúdo", "conceito",
               "defin", "significado", "me fale", "me explique", "me diga sobre"]

PLAN_KW    = ["plano de estudos", "planejamento", "o que estudar", "o que devo",
               "priorizar", "prioridade", "como me preparar", "monte um plano",
               "sequência de estudos", "por onde começar", "me ajude a estudar"]

EXERC_KW   = ["exercício", "exercícios", "questão", "questões", "gere exercício",
               "crie exercício", "me dê exercício", "pratique", "praticar",
               "teste meu conhecimento", "me teste"]

RECALL_KW  = ["active recall", "recall", "me pergunte", "me faça perguntas",
               "iniciar recall", "sessão de perguntas", "quiz", "me questione"]

DIFIC_KW   = ["minhas dificuldades", "onde tenho dificuldade", "o que preciso melhorar",
               "análise do meu desempenho", "identifique minhas dificuldades"]


def _tem(kws, texto):
    return any(k in texto for k in kws)

# ---------------------------------------------------------------------------
# Helpers de extração
# ---------------------------------------------------------------------------

def _extrair_id(msg: str) -> Optional[int]:
    m = re.search(r"#\s*(\d+)|\bnúmero\s+(\d+)|\b(\d+)\b", msg)
    if m:
        val = m.group(1) or m.group(2) or m.group(3)
        return int(val)
    return None


def _extrair_data(texto: str) -> str:
    from datetime import date, timedelta
    t = texto.lower()
    hoje = date.today()
    if "amanhã" in t or "amanha" in t:
        return (hoje + timedelta(days=1)).isoformat()
    if "depois de amanhã" in t or "depois de amanha" in t:
        return (hoje + timedelta(days=2)).isoformat()
    if "hoje" in t:
        return hoje.isoformat()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", texto)
    if m: return m.group(1)
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", texto)
    if m: return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", texto)
    if m: return f"{hoje.year}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return ""


def _extrair_hora(texto: str) -> str:
    m = re.search(r"\b(\d{1,2})[h:](\d{2})?\b", texto)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2) or '00'}"
    return ""


def _extrair_titulo_aspas(msg: str) -> str:
    m = re.search(r'["\']([^"\']+)["\']', msg)
    return m.group(1).strip() if m else ""


def _extrair_titulo_limpo(msg: str, prefixos: list) -> str:
    texto = msg
    for p in sorted(prefixos, key=len, reverse=True):
        texto = re.sub(rf"(?i){re.escape(p)}\s*", "", texto, count=1).strip()
    texto = re.sub(
        r"(?i)\s+(na minha agenda|na agenda|à agenda|a agenda|"
        r"para o dia|no dia|para amanhã|para hoje|de amanhã|de hoje|"
        r"amanhã|amanha|hoje|semana|às|as|para|no dia|em)\b.*$",
        "", texto
    ).strip()
    texto = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", texto)
    texto = re.sub(r"\b\d{1,2}/\d{1,2}(/\d{4})?\b", "", texto)
    texto = re.sub(r"\b\d{1,2}[h:]\d{2}\b", "", texto)
    texto = re.sub(r"\b\d{1,2}h\b", "", texto)
    texto = re.sub(
        r"(?i)\b(minha|meu|um|uma|novo|nova|de|do|da|"
        r"alta|média|media|baixa|prioridade|prazo)\b", "", texto
    )
    return re.sub(r"\s+", " ", texto).strip(' "\',-.')


def _extrair_prioridade(msg: str) -> str:
    if re.search(r"\balta\b",          msg, re.IGNORECASE): return "alta"
    if re.search(r"\bbaixa\b",         msg, re.IGNORECASE): return "baixa"
    return "média"


def _extrair_tipo_evento(msg: str) -> str:
    m = msg.lower()
    for t in ["prova", "aula", "entrega", "reunião", "reuniao"]:
        if t in m:
            return t.replace("reuniao", "reunião")
    return "outro"


def _extrair_tema(msg: str) -> str:
    """Extrai tema após palavras como 'sobre', 'de', 'para'."""
    m = re.search(
        r"(?:sobre|de|para|do tema|da matéria|referente a|acerca de)\s+(.+?)(?:\.|$)",
        msg, re.IGNORECASE
    )
    if m: return m.group(1).strip()
    titulo = _extrair_titulo_aspas(msg)
    if titulo: return titulo
    return msg  # usa a mensagem inteira como tema


PREFIXOS_AGENDA = [
    "adicione evento", "adicione um evento", "adicione uma", "adicione",
    "crie evento", "crie um evento", "crie", "agende", "registre", "insira",
]
PREFIXOS_TAREFA = [
    "adicione tarefa", "adicione uma tarefa", "adicione",
    "crie tarefa", "crie uma tarefa", "crie", "registre", "insira",
]

# ---------------------------------------------------------------------------
# Detecção de intenção
# ---------------------------------------------------------------------------

def _detectar_intencao(msg: str) -> str:
    m = msg.lower()

    # Active recall em andamento — próxima mensagem é a resposta do usuário
    if _active_recall_state["ativo"]:
        return "recall_resposta"

    eh_add  = _tem(ADD_KW,  m)
    eh_edit = _tem(EDIT_KW, m)
    eh_del  = _tem(DEL_KW,  m)
    eh_done = _tem(DONE_KW, m)
    eh_agenda = _tem(AGENDA_KW, m)
    eh_task   = _tem(TASKS_KW,  m)

    # Funcionalidades de aprendizado
    if _tem(RECALL_KW,  m): return "recall_iniciar"
    if _tem(EXERC_KW,   m): return "exercicios"
    if _tem(DIFIC_KW,   m): return "dificuldades"
    if _tem(PLAN_KW,    m): return "planejamento"

    # Agenda
    if eh_agenda:
        if eh_add:  return "agenda_add"
        if eh_edit: return "agenda_edit"
        if eh_del:  return "agenda_del"
        return "agenda_consulta"

    # Tarefas
    if eh_task:
        if eh_add:  return "tarefas_add"
        if eh_edit: return "tarefas_edit"
        if eh_del:  return "tarefas_del"
        if eh_done: return "tarefas_done"
        return "tarefas_list"

    if eh_done: return "tarefas_done"

    if eh_add:
        if _extrair_data(m): return "agenda_add"
        return "tarefas_add"

    if _tem(PERIODO_KW, m): return "agenda_consulta"

    if _tem(RAG_KW, m): return "rag"
    if rag.build_context(msg): return "rag"

    return "geral"

# ---------------------------------------------------------------------------
# Execução das ações
# ---------------------------------------------------------------------------

def _buscar_contexto(msg: str) -> str:
    intencao = _detectar_intencao(msg)
    m = msg.lower()
    print(f"[DEBUG] intenção detectada: {intencao}")

    # ════════════════════════════════════════════════════════════
    # 3.4 PLANEJAMENTO DE ESTUDOS
    # ════════════════════════════════════════════════════════════

    if intencao == "planejamento":
        tema = _extrair_tema(msg)
        ctx_rag = rag.build_context(msg) if tema else ""
        ctx_plan = gerar_contexto_planejamento(tema)

        instrucao = (
            f"Com base nas informações abaixo, crie um plano de estudos detalhado "
            f"e priorizado para o estudante"
            f"{f' com foco em: {tema}' if tema else ''}.\n\n"
            f"O plano deve:\n"
            f"1. Considerar as provas e entregas próximas como prioridade\n"
            f"2. Sugerir uma ordem lógica de estudo\n"
            f"3. Estimar tempo para cada tópico\n"
            f"4. Incluir dicas práticas de estudo\n\n"
            f"{ctx_plan}"
        )
        if ctx_rag:
            instrucao += f"\n\n=== CONTEÚDO DOS MATERIAIS ===\n{ctx_rag}"

        return instrucao

    # ════════════════════════════════════════════════════════════
    # GERAÇÃO DE EXERCÍCIOS
    # ════════════════════════════════════════════════════════════

    if intencao == "exercicios":
        tema = _extrair_tema(msg)
        ctx_rag = rag.build_context(msg)

        if not ctx_rag:
            return ("=== AVISO ===\nNenhum material encontrado para gerar exercícios. "
                    "Adicione PDFs na pasta docs/ e rode --rebuild.")

        qtd = 3
        m_qtd = re.search(r"\b(\d+)\s+exerc", msg, re.IGNORECASE)
        if m_qtd: qtd = int(m_qtd.group(1))

        print(f"[DEBUG] gerando {qtd} exercício(s) sobre '{tema}'")
        return gerar_prompt_exercicios(ctx_rag, tema, qtd)

    # ════════════════════════════════════════════════════════════
    # ACTIVE RECALL — iniciar sessão
    # ════════════════════════════════════════════════════════════

    if intencao == "recall_iniciar":
        tema = _extrair_tema(msg)
        ctx_rag = rag.build_context(msg)

        if not ctx_rag:
            return ("=== AVISO ===\nNenhum material encontrado para o active recall. "
                    "Adicione PDFs na pasta docs/ e rode --rebuild.")

        _active_recall_state["ativo"]     = True
        _active_recall_state["tema"]      = tema
        _active_recall_state["contexto"]  = ctx_rag
        _active_recall_state["historico"] = []

        print(f"[DEBUG] active recall iniciado sobre '{tema}'")
        return gerar_prompt_active_recall(ctx_rag, tema)

    # ════════════════════════════════════════════════════════════
    # ACTIVE RECALL — avaliar resposta do usuário
    # ════════════════════════════════════════════════════════════

    if intencao == "recall_resposta":
        pergunta   = _active_recall_state["pergunta"]
        ctx_rag    = _active_recall_state["contexto"]
        tema       = _active_recall_state["tema"]

        _active_recall_state["historico"].append(msg)

        # Encerrar sessão
        if any(k in m for k in ["encerrar", "parar", "sair", "chega", "finalizar", "próxima", "proxima", "outra pergunta", "nova pergunta", "faça outra", "faca outra"]):
            # Se pediu nova pergunta (não encerrar)
            if any(k in m for k in ["próxima", "proxima", "outra pergunta", "nova pergunta", "faça outra", "faca outra"]):
                nova_pergunta_prompt = gerar_prompt_active_recall(ctx_rag, tema)
                _active_recall_state["pergunta"] = ""
                return nova_pergunta_prompt

            _active_recall_state["ativo"] = False
            historico = _active_recall_state["historico"]
            print("[DEBUG] recall encerrado")
            if len(historico) > 1:
                return gerar_prompt_identificar_dificuldades(historico)
            return "=== SESSÃO ENCERRADA ===\nSessão de active recall finalizada! Bons estudos! 📚"

        # Monta prompt que avalia a resposta E já gera a próxima pergunta
        prompt_avaliacao_e_proxima = f"""Você é um tutor de active recall. Faça duas coisas em sequência:

PARTE 1 — Avalie a resposta do estudante:
Pergunta feita: {pergunta}
Resposta do estudante: {msg}

Use este formato para a avaliação:
✅ Correto / ⚠️ Parcialmente correto / ❌ Incorreto
**Feedback:** (o que acertou e o que errou)
**Resposta completa:** (resposta ideal baseada nos materiais)
**Dica para fixar:** (como memorizar melhor)

PARTE 2 — Faça UMA nova pergunta diferente sobre "{tema}":
- A pergunta deve ser sobre um aspecto diferente do que já foi perguntado
- Termine com: "Sua resposta:"

MATERIAIS DE REFERÊNCIA:
{ctx_rag}

Execute as duas partes agora:"""

        # Reseta a pergunta para ser salva na próxima chamada ao LLM
        _active_recall_state["pergunta"] = ""
        return prompt_avaliacao_e_proxima

    # ════════════════════════════════════════════════════════════
    # IDENTIFICAÇÃO DE DIFICULDADES
    # ════════════════════════════════════════════════════════════

    if intencao == "dificuldades":
        historico = _active_recall_state.get("historico", [])
        if not historico:
            return ("=== AVISO ===\nFaça primeiro uma sessão de active recall para que "
                    "eu possa identificar suas dificuldades. Digite: 'me faça perguntas sobre [tema]'")
        return gerar_prompt_identificar_dificuldades(historico)

    # ════════════════════════════════════════════════════════════
    # AGENDA
    # ════════════════════════════════════════════════════════════

    if intencao == "agenda_add":
        titulo = _extrair_titulo_aspas(msg) or _extrair_titulo_limpo(msg, PREFIXOS_AGENDA)
        if not titulo:
            return ("=== AVISO ===\nNão consegui identificar o título.\n"
                    "Tente: 'adicione evento \"Prova de BD\" amanhã às 17h'")
        ev = agenda_mod.adicionar_evento(
            titulo = titulo,
            data   = _extrair_data(msg),
            tipo   = _extrair_tipo_evento(msg),
            hora   = _extrair_hora(msg),
        )
        print(f"[DEBUG] evento salvo: #{ev['id']} '{ev['titulo']}' data={ev['data']} hora={ev['hora']}")
        return (f"=== EVENTO ADICIONADO ===\n"
                f"✅ Evento #{ev['id']} criado!\n"
                f"• Título: {ev['titulo']}\n"
                f"• Tipo:   {ev['tipo']}\n"
                f"• Data:   {ev['data'] or 'não informada'}\n"
                f"• Hora:   {ev['hora'] or 'não informada'}")

    if intencao == "agenda_edit":
        eid = _extrair_id(msg)
        if not eid:
            return "=== AVISO ===\nInforme o número do evento. Ex: 'edite o evento #2, novo título \"Prova de BD\"'"
        ev = agenda_mod.editar_evento(
            evento_id = eid,
            titulo    = _extrair_titulo_aspas(msg) or None,
            data      = _extrair_data(msg) or None,
            hora      = _extrair_hora(msg) or None,
        )
        if not ev:
            return f"=== AVISO ===\nEvento #{eid} não encontrado."
        return (f"=== EVENTO EDITADO ===\n✅ Evento #{ev['id']} atualizado!\n"
                f"• Título: {ev['titulo']}\n"
                f"• Data:   {ev['data'] or 'não informada'}\n"
                f"• Hora:   {ev['hora'] or 'não informada'}")

    if intencao == "agenda_del":
        eid = _extrair_id(msg)
        if not eid:
            return "=== AVISO ===\nInforme o número do evento. Ex: 'remova o evento #3'"
        ok = agenda_mod.remover_evento(eid)
        return (f"=== EVENTO REMOVIDO ===\n✅ Evento #{eid} removido!" if ok
                else f"=== AVISO ===\nEvento #{eid} não encontrado.")

    if intencao == "agenda_consulta":
        if "amanhã" in m or "amanha" in m:
            eventos, periodo = agenda_mod.eventos_amanha(), "amanhã"
        elif "semana" in m:
            eventos, periodo = agenda_mod.eventos_semana(), "esta semana"
        else:
            eventos, periodo = agenda_mod.eventos_hoje(), "hoje"
        print(f"[DEBUG] agenda ({periodo}): {len(eventos)} evento(s)")
        return (f"=== AGENDA – {periodo.upper()} ===\n"
                f"{agenda_mod.formatar_eventos(eventos)}\n"
                "(Responda com base apenas nesses eventos)")

    # ════════════════════════════════════════════════════════════
    # TAREFAS
    # ════════════════════════════════════════════════════════════

    if intencao == "tarefas_add":
        titulo = _extrair_titulo_aspas(msg) or _extrair_titulo_limpo(msg, PREFIXOS_TAREFA)
        if not titulo:
            return ("=== AVISO ===\nNão consegui identificar o título.\n"
                    "Tente: 'adicione tarefa \"Estudar BD\" para 25/06 prioridade alta'")
        disc_m = re.search(r"\[([^\]]+)\]", msg)
        t = tasks_mod.adicionar_tarefa(
            titulo     = titulo,
            prazo      = _extrair_data(msg),
            disciplina = disc_m.group(1) if disc_m else "",
            prioridade = _extrair_prioridade(msg),
        )
        print(f"[DEBUG] tarefa salva: #{t['id']} '{t['titulo']}'")
        extras = []
        if t["prazo"]:      extras.append(f"prazo: {t['prazo']}")
        if t["disciplina"]: extras.append(f"disciplina: {t['disciplina']}")
        extras.append(f"prioridade: {t['prioridade']}")
        return (f"=== TAREFA ADICIONADA ===\n"
                f"✅ Tarefa #{t['id']} criada: '{t['titulo']}'\n"
                + "\n".join(f"• {e}" for e in extras))

    if intencao == "tarefas_edit":
        tid = _extrair_id(msg)
        if not tid:
            return "=== AVISO ===\nInforme o número da tarefa. Ex: 'edite a tarefa #2, novo título \"Estudar BD\"'"
        nova_prio = None
        if re.search(r"\balta\b",          msg, re.IGNORECASE): nova_prio = "alta"
        elif re.search(r"\bbaixa\b",       msg, re.IGNORECASE): nova_prio = "baixa"
        elif re.search(r"\bmédia\b|media", msg, re.IGNORECASE): nova_prio = "média"
        t = tasks_mod.editar_tarefa(
            task_id    = tid,
            titulo     = _extrair_titulo_aspas(msg) or None,
            prazo      = _extrair_data(msg) or None,
            prioridade = nova_prio,
        )
        if not t:
            return f"=== AVISO ===\nTarefa #{tid} não encontrada."
        return (f"=== TAREFA EDITADA ===\n✅ Tarefa #{t['id']} atualizada!\n"
                f"• Título:     {t['titulo']}\n"
                f"• Prazo:      {t['prazo'] or 'não definido'}\n"
                f"• Prioridade: {t['prioridade']}")

    if intencao == "tarefas_del":
        tid = _extrair_id(msg)
        if not tid:
            return "=== AVISO ===\nInforme o número da tarefa. Ex: 'remova a tarefa #4'"
        ok = tasks_mod.remover_tarefa(tid)
        return (f"=== TAREFA REMOVIDA ===\n✅ Tarefa #{tid} removida!" if ok
                else f"=== AVISO ===\nTarefa #{tid} não encontrada.")

    if intencao == "tarefas_done":
        tid = _extrair_id(msg)
        if not tid:
            return "=== AVISO ===\nInforme o número da tarefa. Ex: 'marca tarefa 3 como concluída'"
        ok = tasks_mod.marcar_concluida(tid)
        return (f"=== TAREFA CONCLUÍDA ===\n✅ Tarefa #{tid} concluída!" if ok
                else f"=== AVISO ===\nTarefa #{tid} não encontrada.")

    if intencao == "tarefas_list":
        apenas_pend = _tem(["pendente","pendentes","aberta","abertas","falta"], m)
        tarefas = tasks_mod.listar_tarefas(apenas_pendentes=apenas_pend)
        print(f"[DEBUG] listar tarefas: {len(tarefas)}")
        return (f"=== LISTA DE TAREFAS ===\n"
                f"{tasks_mod.formatar_tarefas(tarefas)}\n"
                "(Responda com base apenas nessa lista)")

    # ════════════════════════════════════════════════════════════
    # RAG
    # ════════════════════════════════════════════════════════════

    if intencao == "rag":
        contexto = rag.build_context(msg)
        if contexto:
            print("[DEBUG] RAG: trechos encontrados")
            return f"=== MATERIAIS DE ESTUDO ===\n{contexto}"
        print("[DEBUG] RAG: nenhum trecho encontrado")
        return "=== MATERIAIS DE ESTUDO ===\nNenhum trecho relevante encontrado nos documentos indexados."

    print("[DEBUG] resposta geral")
    return ""

# ---------------------------------------------------------------------------
# Chat principal
# ---------------------------------------------------------------------------

def chat(mensagem: str, historico: Optional[List[Dict]] = None) -> str:
    if historico is None:
        historico = []

    contexto = _buscar_contexto(mensagem)

    # Active recall — o contexto JÁ É o prompt completo para o LLM
    if _active_recall_state["ativo"] and contexto:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": contexto})
        response = client.chat.completions.create(model=MODEL, messages=messages)
        resposta = response.choices[0].message.content or ""
        # Salva a resposta do LLM como a pergunta atual (para avaliar na próxima rodada)
        if _active_recall_state["pergunta"] == "":
            _active_recall_state["pergunta"] = resposta
        return resposta

    user_content = (
        f"{contexto}\n\nPergunta do usuário: {mensagem}"
        if contexto else mensagem
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(historico[-10:])
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content or ""