"""
jarvis/src/planejamento.py
==========================
Funcionalidade 3.4 – Planejamento de estudos
Combina agenda + tarefas + materiais RAG para gerar plano personalizado
"""

from datetime import date, timedelta
from src import agenda as agenda_mod
from src import tasks  as tasks_mod


def gerar_contexto_planejamento(tema: str = "") -> str:
    """
    Monta um contexto completo combinando:
    - Eventos da semana (agenda)
    - Tarefas pendentes (tarefas)
    - Tema de estudo (RAG, se informado)
    """
    hoje = date.today()
    partes = []

    # ── Agenda da semana ──────────────────────────────────────────────
    eventos = agenda_mod.eventos_semana()
    provas  = [e for e in eventos if e["tipo"] == "prova"]
    entregas= [e for e in eventos if e["tipo"] == "entrega"]

    partes.append("=== AGENDA DA SEMANA ===")
    partes.append(agenda_mod.formatar_eventos(eventos) if eventos else "Nenhum evento esta semana.")

    if provas:
        partes.append("\n⚠️ PROVAS PRÓXIMAS:")
        for p in provas:
            delta = (date.fromisoformat(p["data"]) - hoje).days
            urgencia = "HOJE" if delta == 0 else f"em {delta} dia(s)"
            partes.append(f"  • {p['titulo']} – {urgencia} ({p['data']})")

    if entregas:
        partes.append("\n📋 ENTREGAS PRÓXIMAS:")
        for e in entregas:
            delta = (date.fromisoformat(e["data"]) - hoje).days
            urgencia = "HOJE" if delta == 0 else f"em {delta} dia(s)"
            partes.append(f"  • {e['titulo']} – {urgencia} ({e['data']})")

    # ── Tarefas pendentes ─────────────────────────────────────────────
    tarefas = tasks_mod.listar_tarefas(apenas_pendentes=True)
    partes.append("\n=== TAREFAS PENDENTES ===")
    partes.append(tasks_mod.formatar_tarefas(tarefas) if tarefas else "Nenhuma tarefa pendente.")

    # ── Data atual ────────────────────────────────────────────────────
    partes.append(f"\n=== DATA ATUAL ===\n{hoje.strftime('%A, %d/%m/%Y')}")

    if tema:
        partes.append(f"\n=== TEMA DE ESTUDO SOLICITADO ===\n{tema}")

    return "\n".join(partes)
