"""
IT Helpdesk MCP Server.

Programa aparte que expone dos tools por MCP (Model Context Protocol):
  - create_support_ticket: crea un ticket de soporte
  - check_ticket_status:   consulta el estado de un ticket

Los tickets viven en un diccionario de Python en memoria (como pide el enunciado),
precargado con data/mcp_tool_data.json. Se pierden al cerrar el servidor.

No hace falta correrlo a mano: app_implementation.py lo arranca solo.
Para probarlo suelto:  python mcp_server.py   (se queda esperando mensajes por stdin)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer  # en mcp 1.x se llamaba FastMCP
from mcp.server.mcpserver.exceptions import ToolError

DATA_FILE = Path(__file__).parent / "data" / "mcp_tool_data.json"

server = MCPServer("IT Helpdesk MCP Server", log_level="WARNING")

# ticket_id -> datos del ticket
TICKETS: dict[str, dict] = {t["ticket_id"]: t for t in json.loads(DATA_FILE.read_text(encoding="utf-8"))}


def next_ticket_id() -> str:
    numbers = [int(tid.split("-")[1]) for tid in TICKETS]
    return f"IT-{max(numbers, default=100) + 1}"


@server.tool()
def create_support_ticket(
    issue: str,
    description: str,
    priority: Literal["Low", "Medium", "High"] = "Medium",
    employee: str = "Not provided",
) -> str:
    """Create a new IT support ticket and return it with its ticket ID.

    Args:
        issue: Short title of the problem, e.g. "Laptop running slowly".
        description: More detail about the problem and what was already tried.
        priority: Low, Medium or High, based on how much it blocks the employee's work.
        employee: Name of the employee reporting the issue, if known.
    """
    ticket = {
        "ticket_id": next_ticket_id(),
        "employee": employee,
        "issue": issue,
        "description": description,
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    TICKETS[ticket["ticket_id"]] = ticket
    return json.dumps(ticket, indent=2)


@server.tool()
def check_ticket_status(ticket_id: str) -> str:
    """Get the current status and details of an existing IT support ticket.

    Args:
        ticket_id: The ticket ID, e.g. "IT-101".
    """
    ticket = TICKETS.get(ticket_id.strip().upper())
    if ticket is None:
        # ToolError = error esperado: MCP regresa el mensaje marcado como error.
        # (Cualquier otra excepción se trata como fallo y el mensaje se esconde.)
        raise ToolError(f"Ticket {ticket_id} not found. Existing tickets: {', '.join(TICKETS)}")
    return json.dumps(ticket, indent=2)


if __name__ == "__main__":
    server.run()  # transporte stdio: habla con el cliente por stdin/stdout
