"""
jarvis/src/tasks.py
===================
Módulo de Tarefas – Funcionalidade 3.3
Armazenamento em data/tasks.json
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

_BASE = Path(__file__).resolve().parent.parent
TASKS_FILE = str(_BASE / "data" / "tasks.json")


def _load() -> List[Dict]:
    p = Path(TASKS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        _save([])
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(tasks: List[Dict]):
    p = Path(TASKS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"[TASKS] salvo em {p} — {len(tasks)} tarefa(s)")


def _next_id(tasks: List[Dict]) -> int:
    return max((t["id"] for t in tasks), default=0) + 1


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def adicionar_tarefa(
    titulo:     str,
    disciplina: str = "",
    prazo:      str = "",
    prioridade: str = "média",
) -> Dict:
    tasks = _load()
    tarefa = {
        "id":         _next_id(tasks),
        "titulo":     titulo,
        "disciplina": disciplina,
        "prazo":      prazo,
        "prioridade": prioridade.lower(),
        "concluida":  False,
        "criada_em":  datetime.now().isoformat(timespec="seconds"),
    }
    tasks.append(tarefa)
    _save(tasks)
    return tarefa


def editar_tarefa(
    task_id:    int,
    titulo:     Optional[str] = None,
    disciplina: Optional[str] = None,
    prazo:      Optional[str] = None,
    prioridade: Optional[str] = None,
) -> Optional[Dict]:
    tasks = _load()
    for t in tasks:
        if t["id"] == task_id:
            if titulo     is not None: t["titulo"]     = titulo
            if disciplina is not None: t["disciplina"] = disciplina
            if prazo      is not None: t["prazo"]      = prazo
            if prioridade is not None: t["prioridade"] = prioridade.lower()
            t["editada_em"] = datetime.now().isoformat(timespec="seconds")
            _save(tasks)
            return t
    return None


def marcar_concluida(task_id: int) -> bool:
    tasks = _load()
    for t in tasks:
        if t["id"] == task_id:
            t["concluida"]    = True
            t["concluida_em"] = datetime.now().isoformat(timespec="seconds")
            _save(tasks)
            return True
    return False


def remover_tarefa(task_id: int) -> bool:
    tasks = _load()
    novas = [t for t in tasks if t["id"] != task_id]
    if len(novas) == len(tasks):
        return False
    _save(novas)
    return True


def buscar_por_id(task_id: int) -> Optional[Dict]:
    for t in _load():
        if t["id"] == task_id:
            return t
    return None


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------

def listar_tarefas(apenas_pendentes: bool = False) -> List[Dict]:
    tasks = _load()
    print(f"[TASKS] total no arquivo: {len(tasks)}")
    if apenas_pendentes:
        tasks = [t for t in tasks if not t["concluida"]]
    ordem = {"alta": 0, "média": 1, "baixa": 2}
    return sorted(
        tasks,
        key=lambda t: (t["concluida"], ordem.get(t["prioridade"], 9), t.get("prazo", "9999")),
    )


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------

def formatar_tarefas(tasks: List[Dict]) -> str:
    if not tasks:
        return "Nenhuma tarefa encontrada."
    ICONS = {"alta": "🔴", "média": "🟡", "baixa": "🟢"}
    linhas = []
    for t in tasks:
        status = "✅" if t["concluida"] else "⬜"
        prio   = ICONS.get(t["prioridade"], "⚪")
        disc   = f" [{t['disciplina']}]" if t.get("disciplina") else ""
        prazo  = f" (até {t['prazo']})"  if t.get("prazo")      else ""
        linhas.append(f"{status} {prio} #{t['id']} {t['titulo']}{disc}{prazo}")
    return "\n".join(linhas)
