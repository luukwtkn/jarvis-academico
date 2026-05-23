"""
jarvis/src/agenda.py
====================
Módulo de Agenda Acadêmica – Funcionalidade 3.2
Armazenamento em data/agenda.json
"""

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# Caminho absoluto relativo ao arquivo para garantir que encontra o JSON
_BASE = Path(__file__).resolve().parent.parent
AGENDA_FILE = str(_BASE / "data" / "agenda.json")


def _load() -> List[Dict]:
    p = Path(AGENDA_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        _save([])
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(eventos: List[Dict]):
    p = Path(AGENDA_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(eventos, f, ensure_ascii=False, indent=2)
    print(f"[AGENDA] salvo em {p} — {len(eventos)} evento(s)")


def _next_id(eventos: List[Dict]) -> int:
    return max((e["id"] for e in eventos), default=0) + 1


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _resolver_data(texto: str) -> str:
    """
    Converte expressões como 'amanhã', 'hoje', 'semana que vem'
    e datas numéricas para formato YYYY-MM-DD.
    """
    t = texto.lower()
    hoje = date.today()

    if "amanhã" in t or "amanha" in t:
        return (hoje + timedelta(days=1)).isoformat()
    if "hoje" in t:
        return hoje.isoformat()
    if "depois de amanhã" in t or "depois de amanha" in t:
        return (hoje + timedelta(days=2)).isoformat()

    import re
    # YYYY-MM-DD
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", texto)
    if m: return m.group(1)
    # DD/MM/YYYY
    m = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", texto)
    if m: return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # DD/MM
    m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", texto)
    if m: return f"{hoje.year}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    return ""


def _extrair_hora(texto: str) -> str:
    import re
    m = re.search(r"\b(\d{1,2})[h:](\d{2})?\b", texto)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2) or '00'}"
    return ""


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def adicionar_evento(
    titulo: str,
    data: str = "",
    tipo: str = "outro",
    hora: str = "",
    local: str = "",
    descricao: str = "",
    **kwargs,
) -> Dict:
    eventos = _load()
    evento = {
        "id":        _next_id(eventos),
        "tipo":      tipo.lower(),
        "titulo":    titulo,
        "data":      data,
        "hora":      hora,
        "local":     local,
        "descricao": descricao,
    }
    eventos.append(evento)
    _save(eventos)
    return evento


def editar_evento(
    evento_id: int,
    titulo:    Optional[str] = None,
    data:      Optional[str] = None,
    tipo:      Optional[str] = None,
    hora:      Optional[str] = None,
    local:     Optional[str] = None,
    descricao: Optional[str] = None,
) -> Optional[Dict]:
    eventos = _load()
    for ev in eventos:
        if ev["id"] == evento_id:
            if titulo    is not None: ev["titulo"]    = titulo
            if data      is not None: ev["data"]      = data
            if tipo      is not None: ev["tipo"]      = tipo.lower()
            if hora      is not None: ev["hora"]      = hora
            if local     is not None: ev["local"]     = local
            if descricao is not None: ev["descricao"] = descricao
            _save(eventos)
            return ev
    return None


def remover_evento(evento_id: int) -> bool:
    eventos = _load()
    novos = [e for e in eventos if e["id"] != evento_id]
    if len(novos) == len(eventos):
        return False
    _save(novos)
    return True


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------

def listar_todos() -> List[Dict]:
    return sorted(_load(), key=lambda e: (e["data"], e.get("hora", "")))


def eventos_hoje() -> List[Dict]:
    hoje = date.today().isoformat()
    todos = _load()
    print(f"[AGENDA] total no arquivo: {len(todos)} | buscando data={hoje}")
    return [e for e in todos if e["data"] == hoje]


def eventos_amanha() -> List[Dict]:
    amanha = (date.today() + timedelta(days=1)).isoformat()
    todos = _load()
    print(f"[AGENDA] total no arquivo: {len(todos)} | buscando data={amanha}")
    return [e for e in todos if e["data"] == amanha]


def eventos_semana() -> List[Dict]:
    hoje = date.today()
    fim  = hoje + timedelta(days=6)
    return [e for e in _load() if e["data"] and hoje <= _parse_date(e["data"]) <= fim]


def buscar_por_id(evento_id: int) -> Optional[Dict]:
    for e in _load():
        if e["id"] == evento_id:
            return e
    return None


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------

def formatar_eventos(eventos: List[Dict]) -> str:
    if not eventos:
        return "Nenhum evento encontrado."
    linhas = []
    for e in sorted(eventos, key=lambda x: (x["data"], x.get("hora", ""))):
        hora  = f" às {e['hora']}"         if e.get("hora")     else ""
        local = f" – {e['local']}"         if e.get("local")    else ""
        desc  = f"\n    Obs: {e['descricao']}" if e.get("descricao") else ""
        linhas.append(
            f"• [#{e['id']}] [{e['tipo'].upper()}] {e['titulo']} – {e['data']}{hora}{local}{desc}"
        )
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Dados de exemplo
# ---------------------------------------------------------------------------

def popular_agenda_exemplo():
    if not _load():
        hoje = date.today()
        exemplos = [
            {"titulo": "Inteligência Artificial", "data": hoje.isoformat(),
             "tipo": "aula", "hora": "08:00", "local": "Sala 301", "descricao": "RAG e LLMs"},
            {"titulo": "Prova de Cálculo II", "data": (hoje + timedelta(days=1)).isoformat(),
             "tipo": "prova", "hora": "14:00", "local": "Auditório B", "descricao": ""},
            {"titulo": "Entrega Trabalho de IA", "data": (hoje + timedelta(days=3)).isoformat(),
             "tipo": "entrega", "hora": "23:59", "local": "", "descricao": "Funcionalidades 3.1-3.3"},
        ]
        for ev in exemplos:
            adicionar_evento(**ev)
        print("[Agenda] Eventos de exemplo inseridos.")
