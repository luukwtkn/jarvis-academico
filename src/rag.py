"""
jarvis/src/rag.py
=================
Módulo RAG (Retrieval-Augmented Generation)
Funcionalidade 3.1 – Consulta a materiais de estudo

Fluxo:
  1. Carrega PDFs / TXTs da pasta DOCS_PATH
  2. Divide em chunks com sobreposição
  3. Gera embeddings via sentence-transformers
  4. Armazena em índice FAISS local
  5. Na consulta: recupera chunks relevantes → monta contexto → envia ao LLM
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple

# Leitura de PDF
import fitz  # PyMuPDF

# Embeddings locais (não precisa de API)
from sentence_transformers import SentenceTransformer

# Busca vetorial
import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Configurações padrão (sobrescritas por variáveis de ambiente / .env)
# ---------------------------------------------------------------------------
DOCS_PATH   = os.getenv("DOCS_PATH",   "./docs")
INDEX_PATH  = os.getenv("INDEX_PATH",  "./data/faiss_index")
EMBED_MODEL = "all-MiniLM-L6-v2"   # modelo leve, ~80 MB, sem GPU

CHUNK_SIZE    = 500   # caracteres por chunk
CHUNK_OVERLAP = 100   # sobreposição entre chunks
TOP_K         = 4     # quantos chunks recuperar por consulta


# ---------------------------------------------------------------------------
# Helpers de chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Divide texto em chunks com sobreposição."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 50]  # descarta chunks muito curtos


def _read_pdf(path: str) -> str:
    """Extrai texto de um PDF usando PyMuPDF."""
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Classe principal do RAG
# ---------------------------------------------------------------------------

class RAGSystem:
    """Gerencia indexação e recuperação de documentos acadêmicos."""

    def __init__(self):
        print("[RAG] Carregando modelo de embeddings...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.chunks: List[str] = []        # textos dos chunks
        self.sources: List[str] = []       # nome do arquivo de origem
        self.index: faiss.IndexFlatL2 | None = None
        self._load_or_build_index()

    # ------------------------------------------------------------------
    # Indexação
    # ------------------------------------------------------------------

    def _load_or_build_index(self):
        """Carrega índice existente ou constrói um novo."""
        idx_file   = INDEX_PATH + ".faiss"
        meta_file  = INDEX_PATH + "_meta.pkl"

        if Path(idx_file).exists() and Path(meta_file).exists():
            print("[RAG] Índice encontrado – carregando...")
            self.index = faiss.read_index(idx_file)
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
            self.chunks  = meta["chunks"]
            self.sources = meta["sources"]
            print(f"[RAG] {len(self.chunks)} chunks carregados de {idx_file}")
        else:
            print("[RAG] Índice não encontrado – construindo do zero...")
            self.build_index()

    def build_index(self):
        """Lê todos os documentos de DOCS_PATH e (re)constrói o índice FAISS."""
        docs_dir = Path(DOCS_PATH)
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RAG] Pasta '{DOCS_PATH}' criada. Coloque seus PDFs/TXTs lá.")
            return

        all_chunks  = []
        all_sources = []

        for file in docs_dir.iterdir():
            suffix = file.suffix.lower()
            if suffix == ".pdf":
                text = _read_pdf(str(file))
            elif suffix in (".txt", ".md"):
                text = _read_txt(str(file))
            else:
                continue  # ignora outros formatos

            chunks = _chunk_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([file.name] * len(chunks))
            print(f"[RAG]   {file.name}: {len(chunks)} chunks")

        if not all_chunks:
            print("[RAG] Nenhum documento encontrado para indexar.")
            return

        # Gera embeddings
        print(f"[RAG] Gerando embeddings para {len(all_chunks)} chunks...")
        embeddings = self.embed_model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Cria índice FAISS
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.chunks  = all_chunks
        self.sources = all_sources

        # Salva no disco
        Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH + ".faiss")
        with open(INDEX_PATH + "_meta.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "sources": self.sources}, f)

        print(f"[RAG] Índice salvo em '{INDEX_PATH}'.")

    def force_rebuild(self):
        """Apaga e reconstrói o índice (use ao adicionar novos documentos)."""
        for ext in (".faiss", "_meta.pkl"):
            p = Path(INDEX_PATH + ext)
            if p.exists():
                p.unlink()
        self.chunks = []
        self.sources = []
        self.index = None
        self.build_index()

    # ------------------------------------------------------------------
    # Recuperação
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, str, float]]:
        """
        Recupera os chunks mais relevantes para a consulta.

        Retorna lista de (chunk_text, source_file, distance).
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], self.sources[idx], float(dist)))
        return results

    def build_context(self, query: str) -> str:
        """
        Monta o bloco de contexto para enviar ao LLM, com trechos recuperados.
        """
        hits = self.retrieve(query)
        if not hits:
            return ""

        parts = []
        for i, (chunk, source, _) in enumerate(hits, 1):
            parts.append(f"[Trecho {i} – {source}]\n{chunk}")

        return "\n\n---\n\n".join(parts)
