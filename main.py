"""
jarvis/main.py
==============
Ponto de entrada do JARVIS Acadêmico
Interface de linha de comando com histórico de conversa

Uso:
  python main.py          → modo chat interativo
  python main.py --rebuild → reconstrói índice RAG e popula agenda de exemplo
"""

import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

# Rich para output bonito no terminal
try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.prompt  import Prompt
    from rich.markdown import Markdown
    USE_RICH = True
except ImportError:
    USE_RICH = False

from src import agenda as agenda_mod
from src.llm import chat

console = Console() if USE_RICH else None


# ---------------------------------------------------------------------------
# Utilitários de display
# ---------------------------------------------------------------------------

def print_banner():
    banner = (
    r"     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗" + "\n"
    r"     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝" + "\n"
    r"     ██║███████║██████╔╝██║   ██║██║███████╗" + "\n"
    r"██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║" + "\n"
    r"╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║" + "\n"
    r" ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝" + "\n"
    "\n"
    "   Assistente Inteligente Acadêmico · Trabalho de IA\n"
)
    if USE_RICH:
        console.print(Panel(banner, style="bold cyan", subtitle="Digite 'sair' para encerrar"))
    else:
        print(banner)
        print("=" * 60)
        print("Digite 'sair' para encerrar | 'ajuda' para ver comandos")
        print("=" * 60)


def print_ajuda():
    ajuda = """
**Comandos especiais:**
- `sair` / `exit`         → encerra o JARVIS
- `limpar`                → limpa o histórico de conversa
- `rebuild`               → reconstrói o índice RAG com novos documentos
- `agenda popular`        → popula agenda com eventos de exemplo
- `ajuda`                 → mostra esta mensagem

**Exemplos de perguntas:**
- "O que tenho hoje?"
- "Tenho prova amanhã?"
- "Explique regressão logística"
- "Adiciona tarefa: estudar para prova de cálculo, prazo 2025-06-15, prioridade alta"
- "Lista minhas tarefas pendentes"
- "Marca tarefa 3 como concluída"
"""
    if USE_RICH:
        console.print(Panel(Markdown(ajuda), title="Ajuda", style="yellow"))
    else:
        print(ajuda)


def print_resposta(texto: str):
    if USE_RICH:
        console.print(Panel(Markdown(texto), title="🤖 JARVIS", style="green"))
    else:
        print(f"\n[JARVIS] {texto}\n")


def print_erro(texto: str):
    if USE_RICH:
        console.print(f"[bold red]Erro:[/bold red] {texto}")
    else:
        print(f"ERRO: {texto}")


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def run_chat():
    print_banner()
    historico = []

    while True:
        try:
            if USE_RICH:
                user_input = Prompt.ask("\n[bold blue]Você[/bold blue]").strip()
            else:
                user_input = input("\nVocê: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAté logo! 👋")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # Comandos locais (sem chamar o LLM)
        if cmd in ("sair", "exit", "quit"):
            if USE_RICH:
                console.print("[bold cyan]Até logo! 👋[/bold cyan]")
            else:
                print("Até logo! 👋")
            break

        elif cmd == "ajuda":
            print_ajuda()
            continue

        elif cmd == "limpar":
            historico = []
            if USE_RICH:
                console.print("[italic]Histórico limpo.[/italic]")
            else:
                print("Histórico limpo.")
            continue

        elif cmd == "rebuild":
            from src.rag import RAGSystem
            print("Reconstruindo índice RAG...")
            r = RAGSystem()
            r.force_rebuild()
            print("Índice reconstruído!")
            continue

        elif cmd == "agenda popular":
            agenda_mod.popular_agenda_exemplo()
            print("Agenda de exemplo inserida!")
            continue

        # Chama o LLM
        try:
            if USE_RICH:
                with console.status("[cyan]JARVIS pensando...[/cyan]"):
                    resposta = chat(user_input, historico)
            else:
                print("JARVIS pensando...")
                resposta = chat(user_input, historico)

            print_resposta(resposta)

            # Atualiza histórico (mantém últimas 10 trocas para não explodir o contexto)
            historico.append({"role": "user",      "content": user_input})
            historico.append({"role": "assistant", "content": resposta})
            if len(historico) > 20:
                historico = historico[-20:]

        except Exception as e:
            print_erro(f"{type(e).__name__}: {e}")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Argumentos de linha de comando
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JARVIS Acadêmico")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstrói o índice RAG e popula agenda de exemplo, depois encerra.",
    )
    args = parser.parse_args()

    if args.rebuild:
        print("=== Setup inicial do JARVIS ===")
        agenda_mod.popular_agenda_exemplo()
        from src.rag import RAGSystem
        r = RAGSystem()
        r.force_rebuild()
        print("Pronto! Execute 'python main.py' para iniciar o chat.")
        return

    run_chat()


if __name__ == "__main__":
    main()
