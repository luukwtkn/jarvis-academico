"""
jarvis/src/llm.py
=================
Cliente LLM – integração com Gemma 12B
Baseado no v6 (RAG funcional) + CRUD completo de agenda e tarefas
"""

import os
import re
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

from src.rag    import RAGSystem
from src import agenda as agenda_mod
from src import tasks  as tasks_mod

load_dotenv()

client = OpenAI(
    api_key  = os.getenv("GEMMA_API_KEY", ""),
    base_url = os.getenv("GEMMA_BASE_URL", "https://api.example.com/v1"),
)
MODEL = os.getenv("GEMMA_MODEL", "gemma-12b")

rag = RAGSystem()

SYSTEM_PROMPT = """Você é o JARVIS Acadêmico, um assistente pessoal inteligente para estudantes universitários.

Quando houver informações de contexto fornecidas (dentro de blocos ===), use APENAS essas informações.
Não invente eventos, tarefas ou conteúdos que não estejam no contexto.
Se uma ação foi realizada (adicionar, editar, remover), confirme de forma clara ao usuário.
Responda sempre em português brasileiro de forma direta e amigável.
"""

# ---------------------------------------------------------------------------
# Palavras-chave — igual ao v6, expandido para CRUD
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

# RAG — igual ao v6
RAG_KW = ["explique", "explica", "resumo", "resuma", "o que é", "o que são",
           "como funciona", "fale sobre", "material", "conteúdo", "conceito",
           "defin", "significado", "me fale", "me explique", "me diga sobre",
           "sobre o tema", "sobre o assunto"]


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
    """Converte amanhã/hoje e formatos DD/MM para YYYY-MM-DD."""
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
    """Remove prefixo de comando e referências de tempo para isolar o título."""
    texto = msg
    for p in sorted(prefixos, key=len, reverse=True):
        texto = re.sub(rf"(?i){re.escape(p)}\s*", "", texto, count=1).strip()
    # Corta a partir de palavras temporais/locais
    texto = re.sub(
        r"(?i)\s+(na minha agenda|na agenda|à agenda|a agenda|"
        r"para o dia|no dia|para amanhã|para hoje|de amanhã|de hoje|"
        r"amanhã|amanha|hoje|semana|às|as|para|no dia|em)\b.*$",
        "", texto
    ).strip()
    # Remove datas e horas residuais
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


PREFIXOS_AGENDA = [
    "adicione evento", "adicione um evento", "adicione uma", "adicione",
    "crie evento", "crie um evento", "crie", "agende", "registre", "insira",
]
PREFIXOS_TAREFA = [
    "adicione tarefa", "adicione uma tarefa", "adicione",
    "crie tarefa", "crie uma tarefa", "crie", "registre", "insira",
]

# ---------------------------------------------------------------------------
# Detecção de intenção — mesma lógica do v6, agora com CRUD
# ---------------------------------------------------------------------------

def _detectar_intencao(msg: str) -> str:
    m = msg.lower()

    eh_add  = _tem(ADD_KW,  m)
    eh_edit = _tem(EDIT_KW, m)
    eh_del  = _tem(DEL_KW,  m)
    eh_done = _tem(DONE_KW, m)
    eh_agenda = _tem(AGENDA_KW, m)
    eh_task   = _tem(TASKS_KW,  m)

    # Agenda explícita
    if eh_agenda:
        if eh_add:  return "agenda_add"
        if eh_edit: return "agenda_edit"
        if eh_del:  return "agenda_del"
        return "agenda_consulta"

    # Tarefa explícita
    if eh_task:
        if eh_add:  return "tarefas_add"
        if eh_edit: return "tarefas_edit"
        if eh_del:  return "tarefas_del"
        if eh_done: return "tarefas_done"
        return "tarefas_list"

    # Verbo de conclusão sozinho → tarefa
    if eh_done: return "tarefas_done"

    # Verbo de adição sem keyword → infere pelo contexto
    if eh_add:
        if _extrair_data(m): return "agenda_add"
        return "tarefas_add"

    # Consulta por período → agenda
    if _tem(PERIODO_KW, m): return "agenda_consulta"

    # RAG — igual ao v6: verifica keywords E tenta busca vetorial
    if _tem(RAG_KW, m): return "rag"
    contexto_rag = rag.build_context(msg)
    if contexto_rag: return "rag"

    return "geral"

# ---------------------------------------------------------------------------
# Execução das ações
# ---------------------------------------------------------------------------

def _buscar_contexto(msg: str) -> str:
    intencao = _detectar_intencao(msg)
    m = msg.lower()
    print(f"[DEBUG] intenção detectada: {intencao}")

    # ── Agenda: adicionar ────────────────────────────────────────────────
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

    # ── Agenda: editar ───────────────────────────────────────────────────
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
        print(f"[DEBUG] evento editado: #{eid}")
        return (f"=== EVENTO EDITADO ===\n✅ Evento #{ev['id']} atualizado!\n"
                f"• Título: {ev['titulo']}\n"
                f"• Data:   {ev['data'] or 'não informada'}\n"
                f"• Hora:   {ev['hora'] or 'não informada'}")

    # ── Agenda: remover ──────────────────────────────────────────────────
    if intencao == "agenda_del":
        eid = _extrair_id(msg)
        if not eid:
            return "=== AVISO ===\nInforme o número do evento. Ex: 'remova o evento #3'"
        ok = agenda_mod.remover_evento(eid)
        print(f"[DEBUG] remover evento #{eid}: {ok}")
        return (f"=== EVENTO REMOVIDO ===\n✅ Evento #{eid} removido!" if ok
                else f"=== AVISO ===\nEvento #{eid} não encontrado.")

    # ── Agenda: consultar ────────────────────────────────────────────────
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

    # ── Tarefas: adicionar ───────────────────────────────────────────────
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
        print(f"[DEBUG] tarefa salva: #{t['id']} '{t['titulo']}' prazo={t['prazo']}")
        extras = []
        if t["prazo"]:      extras.append(f"prazo: {t['prazo']}")
        if t["disciplina"]: extras.append(f"disciplina: {t['disciplina']}")
        extras.append(f"prioridade: {t['prioridade']}")
        return (f"=== TAREFA ADICIONADA ===\n"
                f"✅ Tarefa #{t['id']} criada: '{t['titulo']}'\n"
                + "\n".join(f"• {e}" for e in extras))

    # ── Tarefas: editar ──────────────────────────────────────────────────
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
        print(f"[DEBUG] tarefa editada: #{tid}")
        return (f"=== TAREFA EDITADA ===\n✅ Tarefa #{t['id']} atualizada!\n"
                f"• Título:     {t['titulo']}\n"
                f"• Prazo:      {t['prazo'] or 'não definido'}\n"
                f"• Prioridade: {t['prioridade']}")

    # ── Tarefas: remover ─────────────────────────────────────────────────
    if intencao == "tarefas_del":
        tid = _extrair_id(msg)
        if not tid:
            return "=== AVISO ===\nInforme o número da tarefa. Ex: 'remova a tarefa #4'"
        ok = tasks_mod.remover_tarefa(tid)
        print(f"[DEBUG] remover tarefa #{tid}: {ok}")
        return (f"=== TAREFA REMOVIDA ===\n✅ Tarefa #{tid} removida!" if ok
                else f"=== AVISO ===\nTarefa #{tid} não encontrada.")

    # ── Tarefas: concluir ────────────────────────────────────────────────
    if intencao == "tarefas_done":
        tid = _extrair_id(msg)
        if not tid:
            return "=== AVISO ===\nInforme o número da tarefa. Ex: 'marca tarefa 3 como concluída'"
        ok = tasks_mod.marcar_concluida(tid)
        print(f"[DEBUG] concluir #{tid}: {ok}")
        return (f"=== TAREFA CONCLUÍDA ===\n✅ Tarefa #{tid} concluída!" if ok
                else f"=== AVISO ===\nTarefa #{tid} não encontrada.")

    # ── Tarefas: listar ──────────────────────────────────────────────────
    if intencao == "tarefas_list":
        apenas_pend = _tem(["pendente","pendentes","aberta","abertas","falta"], m)
        tarefas = tasks_mod.listar_tarefas(apenas_pendentes=apenas_pend)
        print(f"[DEBUG] listar tarefas: {len(tarefas)} encontrada(s)")
        return (f"=== LISTA DE TAREFAS ===\n"
                f"{tasks_mod.formatar_tarefas(tarefas)}\n"
                "(Responda com base apenas nessa lista)")

    # ── RAG — lógica idêntica ao v6 ──────────────────────────────────────
    if intencao == "rag":
        contexto = rag.build_context(msg)
        if contexto:
            print("[DEBUG] RAG: trechos encontrados")
            return f"=== MATERIAIS DE ESTUDO ===\n{contexto}"
        print("[DEBUG] RAG: nenhum trecho encontrado")
        return "=== MATERIAIS DE ESTUDO ===\nNenhum trecho relevante encontrado nos documentos indexados."

    print("[DEBUG] resposta geral (sem contexto)")
    return ""


# ---------------------------------------------------------------------------
# Chat principal — idêntico ao v6
# ---------------------------------------------------------------------------

def chat(mensagem: str, historico: Optional[List[Dict]] = None) -> str:
    if historico is None:
        historico = []

    contexto = _buscar_contexto(mensagem)
    user_content = (
        f"{contexto}\n\nPergunta do usuário: {mensagem}"
        if contexto else mensagem
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(historico[-10:])
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model    = MODEL,
        messages = messages,
    )
    return response.choices[0].message.content or ""
