"""
Interfaz Streamlit del IT Helpdesk Assistant.

Correr desde esta carpeta:
    streamlit run streamlit_app.py

Toda la lógica (Claude, tools, RAG, MCP) vive en app_implementation.py.
"""

import json

import streamlit as st

from app_implementation import run_agent

st.set_page_config(page_title="IT Helpdesk Assistant", page_icon=":material/support_agent:")

# Prompts de ejemplo del enunciado (usecase_Itsupport.pdf)
SUGGESTIONS = {
    ":material/wifi_off: Network issues": "How do I fix network connectivity issues?",
    ":material/speed: Slow computer": "My computer is running slow. What should I do?",
    ":material/mail: Configure email": "How do I configure my email?",
    ":material/shield: Malware protection": "How can I protect my laptop from malware?",
    ":material/schedule: Current date and time": "What is the current date and time?",
    ":material/travel_explore: Windows 11 Wi-Fi fixes": "Search for the latest Windows 11 Wi-Fi troubleshooting steps.",
    ":material/travel_explore: Outlook not syncing": "Find solutions for Outlook not syncing emails.",
    ":material/confirmation_number: Create a ticket": "Create a ticket for my laptop running slowly.",
    ":material/fact_check: Ticket IT-101 status": "Check the status of ticket IT-101.",
}

TOOL_LABELS = {
    "search_it_faq": ":material/menu_book: Searched the IT FAQ",
    "get_current_datetime": ":material/schedule: Checked the date and time",
    "web_search": ":material/travel_explore: Searched the web",
    "create_support_ticket": ":material/confirmation_number: Created a ticket (MCP)",
    "check_ticket_status": ":material/fact_check: Checked a ticket (MCP)",
}


def show_steps(steps: list):
    """Dibuja cada llamada a tool como un paso de la línea de tiempo."""
    for step in steps:
        label = TOOL_LABELS.get(step["tool"], f":material/build: {step['tool']}")
        with st.status(label, type="step", state="error" if step["error"] else "complete"):
            if step["input"]:
                st.caption("Input")
                st.code(json.dumps(step["input"], indent=2, ensure_ascii=False), language="json")
            st.caption("Result")
            st.code(step["output"], language="json")


# --- Estado: se inicializa una sola vez por sesión ---
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []  # historial completo que se manda a Claude
    st.session_state.chat = []          # lo que se dibuja en pantalla

with st.sidebar:
    st.header("IT Helpdesk")
    st.caption("Ask about passwords, Wi-Fi, email, software, laptops or security.")
    if st.button(":material/delete: New conversation", width="stretch"):
        st.session_state.api_messages = []
        st.session_state.chat = []
        st.rerun()

st.title(":material/support_agent: IT Helpdesk Assistant")

# --- Historial ---
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        if msg.get("steps"):
            with st.expander(f"Used {len(msg['steps'])} tool(s)", type="compact"):
                show_steps(msg["steps"])
        st.markdown(msg["content"])

# --- Entrada: caja de texto o sugerencias ---
prompt = st.chat_input("Describe your IT issue", submit_mode="disable")
if not st.session_state.chat:
    selected = st.pills("Try asking:", list(SUGGESTIONS), label_visibility="collapsed")
    if selected:
        prompt = SUGGESTIONS[selected]

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Copia del historial: si algo falla, el original queda intacto
    messages = st.session_state.api_messages + [{"role": "user", "content": prompt}]

    with st.chat_message("assistant"):
        try:
            with st.status(":shimmer[Working on it]", type="compact") as status:
                answer, steps = run_agent(messages)
                show_steps(steps)
                status.update(
                    label=f"Used {len(steps)} tool(s)" if steps else "Answered directly",
                    state="complete",
                )
            st.markdown(answer)
        except Exception as e:
            st.error(f"Something went wrong: {e}", title="Error")
            st.session_state.chat.pop()  # quita la pregunta para no dejar el historial cojo
            st.stop()

    st.session_state.api_messages = messages
    st.session_state.chat.append({"role": "assistant", "content": answer, "steps": steps})
    st.rerun()  # vuelve a dibujar desde el historial (y oculta las sugerencias)
