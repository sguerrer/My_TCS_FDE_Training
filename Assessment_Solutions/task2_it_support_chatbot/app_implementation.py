"""
Lógica del IT Helpdesk Assistant (sin interfaz).

streamlit_app.py importa de aquí. Este archivo no sabe nada de Streamlit,
así que se puede probar solo desde la consola:  python app_implementation.py

Tools:
  - search_it_faq          (cliente: RAG sobre data/it_sector.txt con Voyage AI + ChromaDB)
  - get_current_datetime   (cliente: corre en este programa)
  - web_search             (servidor: la ejecuta Anthropic)
  - create_support_ticket  (MCP: las ejecuta mcp_server.py)
  - check_ticket_status    (MCP)
"""

import asyncio
import json
import re
import sys
import threading
from contextlib import AsyncExitStack
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import anthropic
import chromadb
import voyageai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "data" / "it_sector.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBED_MODEL = "voyage-3.5-lite"  # el MISMO modelo para guardar y para buscar (serie 3: tokens gratis)
TOP_K = 3

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000

SYSTEM_PROMPT = """You are an IT Helpdesk Assistant for company employees.
Help with IT issues: passwords, Wi-Fi and network, email, software, laptops and security.
Give clear, numbered troubleshooting steps when relevant.
Use your tools whenever they help; never invent dates or ticket data.
Create a support ticket when the employee asks for one, or offer one when the
issue can't be solved with troubleshooting. Always show the ticket ID and its details.
For any IT question, first call search_it_faq to check the company IT FAQ and base
your answer on it. Use web_search when the FAQ doesn't cover the issue, when the user asks for the latest information or for solutions
beyond common IT knowledge, and cite the sources you used.
Reply in the same language the employee writes in."""

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# RAG: preparación (una vez) y búsqueda (en cada pregunta)
# ---------------------------------------------------------------------------
def load_faq_chunks(path: Path) -> list[dict]:
    """
    Divide el FAQ en pedazos. El archivo ya viene ordenado como
    "## Tema / Q: pregunta / A: respuesta", así que cada sección es un pedazo
    completo y con sentido (mejor que cortar cada N caracteres).
    """
    chunks = []
    for section in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M):
        if not section.strip():
            continue
        topic, _, body = section.partition("\n")
        question = re.search(r"Q:\s*(.+)", body)
        chunks.append({
            "id": f"faq-{len(chunks) + 1}",
            "text": f"{topic.strip()}\n{body.strip()}",
            "metadata": {"topic": topic.strip(), "question": question.group(1).strip() if question else ""},
        })
    return chunks


voyage = voyageai.Client()  # lee VOYAGE_API_KEY del .env
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))  # la base vectorial se guarda en disco


def build_index() -> chromadb.Collection:
    """Fase de preparación: cargar -> dividir -> embeddings -> guardar en ChromaDB."""
    collection = chroma.get_or_create_collection(
        # El modelo va en el nombre: si cambias de modelo se crea un índice nuevo
        # en vez de mezclar vectores de modelos distintos (la búsqueda fallaría)
        name=f"it_faq_{EMBED_MODEL.replace('.', '_')}",
        embedding_function=None,          # los embeddings los damos nosotros (Voyage)
        metadata={"hnsw:space": "cosine"},  # parecido medido con similitud coseno
    )
    chunks = load_faq_chunks(FAQ_PATH)
    if collection.count() != len(chunks):  # solo se reconstruye si el FAQ cambió de tamaño
        texts = [c["text"] for c in chunks]
        embeddings = voyage.embed(texts, model=EMBED_MODEL, input_type="document").embeddings
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=texts,               # junto a cada vector se guarda el texto
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )
    return collection


faq_collection = build_index()


@lru_cache(maxsize=256)
def embed_query(query: str) -> tuple[float, ...]:
    """Embedding de la pregunta. Con caché: la misma pregunta no vuelve a llamar a Voyage."""
    return tuple(voyage.embed([query], model=EMBED_MODEL, input_type="query").embeddings[0])


def search_it_faq(query: str) -> str:
    """Fase de ejecución: pregunta -> embedding -> los pedazos más parecidos."""
    try:
        query_embedding = list(embed_query(query))
    except voyageai.error.RateLimitError:
        # Sin método de pago Voyage permite 3 peticiones/minuto. Se le avisa a Claude
        # (como error de la tool) para que lo diga o use otra fuente, sin tronar la app.
        raise RuntimeError("the FAQ search is rate-limited right now; try again in a minute")
    found = faq_collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    results = [
        {"topic": meta["topic"], "similarity": round(1 - dist, 3), "content": doc}
        for doc, meta, dist in zip(found["documents"][0], found["metadatas"][0], found["distances"][0])
    ]
    return json.dumps(results, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MCP: conexión con mcp_server.py
# ---------------------------------------------------------------------------
class MCPConnection:
    """
    Arranca mcp_server.py como proceso hijo y mantiene la conexión abierta.

    El cliente MCP es async y Streamlit no, así que la conexión vive en un
    event loop dentro de un hilo aparte; call_tool() le manda trabajo y espera.
    La conexión NO se reabre por pregunta: si se reabriera, el servidor
    arrancaría de cero y los tickets creados (en memoria) se perderían.
    """

    def __init__(self, server_script: Path):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.tools = self._run(self._connect(server_script))  # definiciones en formato Claude
        self.tool_names = {t["name"] for t in self.tools}

    def _run(self, coroutine, timeout: float = 60):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    async def _connect(self, server_script: Path) -> list[dict]:
        self._stack = AsyncExitStack()  # mantiene abiertos el proceso y la sesión
        params = StdioServerParameters(command=sys.executable, args=[str(server_script)])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        # Las tools MCP traducidas al formato que espera Claude
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in (await self.session.list_tools()).tools
        ]

    def call_tool(self, name: str, args: dict) -> tuple[str, bool]:
        result = self._run(self.session.call_tool(name, args))
        text = "\n".join(c.text for c in result.content if c.type == "text")
        return text, bool(result.is_error)


mcp_client = MCPConnection(BASE_DIR / "mcp_server.py")


# ---------------------------------------------------------------------------
# Tools del cliente: corren en NUESTRO programa
# ---------------------------------------------------------------------------
def get_current_datetime() -> str:
    now = datetime.now().astimezone()
    return json.dumps({
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.strftime("%A, %d %B %Y"),
        "time": now.strftime("%H:%M"),
        "timezone": str(now.tzinfo),
    })


# Lo que Claude ve de cada tool: nombre, descripción y qué parámetros recibe.
# La descripción es lo más importante: con ella decide cuándo usarla.
CLIENT_TOOLS = [
    {
        "name": "search_it_faq",
        "description": "Search the company IT support FAQ (network, software installation, "
                       "performance, backups, email, security/malware, hardware, VPN, mobile devices). "
                       "Returns the most relevant FAQ entries with a similarity score (0-1).",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The IT issue to look up"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current local date and time. Use it whenever the user asks "
                       "for the date or time, or when a date is needed (e.g. for a ticket).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# nombre de la tool -> función de Python que la ejecuta
TOOL_FUNCTIONS = {
    "search_it_faq": lambda args: search_it_faq(args["query"]),
    "get_current_datetime": lambda args: get_current_datetime(),
}


# ---------------------------------------------------------------------------
# Tools del servidor: las ejecuta Anthropic, nosotros solo las declaramos
# ---------------------------------------------------------------------------
# Haiku 5.5 usa la versión básica de web search (la _20260209 es para Sonnet/Opus 5.5).
# Cada búsqueda tiene costo aparte de los tokens, por eso max_uses limita cuántas hace por turno.
SERVER_TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
]


def log_server_tools(content: list, log) -> None:
    """Las tools del servidor ya vienen resueltas en la respuesta: solo las registramos."""
    queries = {b.id: b.input for b in content if b.type == "server_tool_use"}
    for block in content:
        if block.type != "web_search_tool_result":
            continue
        query = queries.get(block.tool_use_id, {})
        if isinstance(block.content, list):  # éxito: lista de resultados
            sources = [{"title": r.title, "url": r.url} for r in block.content]
            log({"tool": "web_search", "input": query, "output": json.dumps(sources, indent=2), "error": False})
        else:  # error: un objeto con error_code (no lanza excepción)
            log({"tool": "web_search", "input": query, "output": f"Error: {block.content.error_code}", "error": True})


ALL_TOOLS = CLIENT_TOOLS + mcp_client.tools + SERVER_TOOLS


def execute_tool(name: str, args: dict) -> tuple[str, bool]:
    """Ejecuta una tool del cliente o de MCP. Regresa (resultado, es_error)."""
    try:
        if name in mcp_client.tool_names:
            return mcp_client.call_tool(name, args)  # la ejecuta el servidor MCP
        return TOOL_FUNCTIONS[name](args), False
    except KeyError:
        return f"Error: unknown tool '{name}'", True
    except Exception as e:  # el error se le pasa a Claude para que lo explique
        return f"Error running {name}: {e}", True


# ---------------------------------------------------------------------------
# Ciclo de tool use
# ---------------------------------------------------------------------------
def run_agent(messages: list, on_step=None) -> tuple[str, list]:
    """
    Manda la conversación a Claude y repite mientras pida tools.

    messages: historial completo (se modifica: se le agregan los turnos nuevos).
    on_step:  función opcional que se llama con cada evento, para mostrarlo en la UI.
    Regresa (texto_final, pasos).
    """
    steps = []

    def log(step: dict):
        steps.append(step)
        if on_step:
            on_step(step)

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            messages=messages,
        )
        # Se guarda TODO el contenido (texto, thinking, tool_use, búsquedas) tal cual llega
        messages.append({"role": "assistant", "content": response.content})
        log_server_tools(response.content, log)

        if response.stop_reason == "pause_turn":
            continue  # una tool del servidor tardó: se reenvía para que Claude continúe

        if response.stop_reason != "tool_use":
            break  # end_turn, max_tokens o refusal: ya no pide tools

        # Claude pidió una o varias tools: las ejecutamos todas
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output, is_error = execute_tool(block.name, block.input)
            log({"tool": block.name, "input": block.input, "output": output, "error": is_error})
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # tiene que coincidir con el id del tool_use
                "content": output,
                "is_error": is_error,
            })
        # Todos los resultados van juntos en UN mensaje de usuario
        messages.append({"role": "user", "content": results})

    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "refusal":
        text = "Sorry, I can't help with that request."
    elif response.stop_reason == "max_tokens":
        text += "\n\n_(Response cut off: reached max_tokens.)_"
    return text, steps


# Prueba rápida desde consola
if __name__ == "__main__":
    history = [{"role": "user", "content": "What is the current date and time?"}]
    answer, steps = run_agent(history, on_step=lambda s: print(f"  [tool] {s['tool']} -> {s['output']}"))
    print(answer)
