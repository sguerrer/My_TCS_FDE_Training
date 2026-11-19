# AI-Powered IT Helpdesk Assistant

A Streamlit chatbot that helps employees with IT issues. Claude (Anthropic API) decides
on its own which tool to use for each question:

| Capability | Tool | Where it runs |
|---|---|---|
| Answer from the IT FAQ (RAG) | `search_it_faq` | This app: Voyage AI embeddings + ChromaDB vector database |
| Current date and time | `get_current_datetime` | This app (client tool) |
| Search the internet | `web_search` | Anthropic's servers (server tool) |
| Create a support ticket | `create_support_ticket` | `mcp_server.py` (IT Helpdesk MCP Server) |
| Check a ticket's status | `check_ticket_status` | `mcp_server.py` (IT Helpdesk MCP Server) |

## Architecture

```
streamlit_app.py  ──►  app_implementation.py  ──►  Claude (claude-haiku-5-5)
   (chat UI)            (tool-use loop)                  │ asks for tools
                              │                          ▼
                              ├── search_it_faq ──► Voyage AI + ChromaDB (data/it_sector.txt)
                              ├── get_current_datetime
                              ├── web_search (runs inside Anthropic, results come back in the response)
                              └── MCP client ──stdio──► mcp_server.py (tickets in an in-memory dict)
```

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Chat interface. Shows each tool call as a step, with its input and result |
| `app_implementation.py` | Claude client, tool definitions, RAG, MCP client and the tool-use loop |
| `mcp_server.py` | IT Helpdesk MCP Server with the two ticket tools |
| `data/it_sector.txt` | IT support FAQ used for RAG |
| `data/mcp_tool_data.json` | Initial tickets (IT-101 to IT-103) loaded into the MCP server |
| `requirements.txt` | Dependencies |

> The UI file is named `streamlit_app.py` instead of `streamlit.py`: a file named like a
> library can shadow it, so `import streamlit` could load this file instead of the package.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env            # then add your ANTHROPIC_API_KEY and VOYAGE_API_KEY
streamlit run streamlit_app.py
```

There is no need to start the MCP server separately: the app launches `mcp_server.py`
automatically and keeps the connection open, so tickets created during a session are kept.
The vector index (`chroma_db/`) is built automatically on the first run.

## Sample prompts

| Category | Prompt | Expected behavior |
|---|---|---|
| RAG | How do I fix network connectivity issues? | `search_it_faq` |
| RAG | My computer is running slow. What should I do? | `search_it_faq` |
| RAG | How do I configure my email? | `search_it_faq` |
| RAG | How can I protect my laptop from malware? | `search_it_faq` |
| Web search | Search for the latest Windows 11 Wi-Fi troubleshooting steps. | `web_search` |
| Web search | Find solutions for Outlook not syncing emails. | `web_search` |
| MCP | Create a ticket for my laptop running slowly. | `create_support_ticket` |
| MCP | Check the status of ticket IT-101. | `check_ticket_status` |
| Tool calling | What is the current date and time? | `get_current_datetime` |

All of them are available as clickable suggestions when the chat is empty.

## Design notes

- **Chunking by FAQ entry.** `it_sector.txt` is already organized as `## Topic / Q: / A:`,
  so each section becomes one chunk instead of splitting every N characters.
- **RAG as a tool.** Claude calls the FAQ search only when the question needs it.
- **Same embedding model to index and to query** (`voyage-3.5-lite`). The model name is part
  of the collection name, so changing models builds a new index instead of mixing vectors.
- **Tool errors don't crash the app.** They are sent back to Claude with `is_error: true`
  (or as an MCP error result), and Claude explains the problem to the user.
- **Voyage free tier** allows 3 requests per minute without a payment method; repeated queries
  are cached, and a rate-limit error is reported to Claude as a tool error.
