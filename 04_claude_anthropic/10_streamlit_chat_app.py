"""
Chat con Claude en Streamlit (tarea Day-10).

Correr desde la raíz del repo, con el .venv activo:
    streamlit run 04_claude_anthropic/10_streamlit_chat_app.py

Proveedores:
  - Claude (Anthropic): el bueno. Necesita ANTHROPIC_API_KEY válida en el .env.
  - Ollama (local): para probar mientras no hay key. No soporta "effort".
  - Modo de prueba: no llama a ningún modelo; muestra la petición que se mandaría.
"""

import json

import anthropic
import ollama
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # busca el .env subiendo carpetas hasta la raíz del repo

CLAUDE_MODELS = {
    "Claude Haiku 5.5 (el más barato)": "claude-haiku-5-5",
    "Claude Sonnet 5.5": "claude-sonnet-5-5",
    "Claude Opus 5.5 (el más caro)": "claude-opus-5-5",
}
EFFORTS = ["low", "medium", "high", "xhigh", "max"]
EFFORTS_SIN_THINKING = ["low", "medium", "high"]  # apagar thinking solo se vale hasta "high"
MAX_TOKENS = 16000

st.set_page_config(page_title="Chat con Claude", page_icon="💬")


# ---------------------------------------------------------------------------
# Armado de la petición (la parte "build the request dynamically")
# ---------------------------------------------------------------------------
def config_thinking(model: str, thinking_on: bool) -> dict:
    """Cada modelo 5.5 apaga el thinking de forma distinta."""
    if thinking_on:
        # "summarized" para poder mostrar el razonamiento; por defecto llega vacío
        return {"type": "adaptive", "display": "summarized"}
    if "opus" in model:
        return {"type": "adaptive"}  # Opus 5.5 no permite apagarlo
    if "sonnet" in model:
        return {"type": "between_tools"}  # en Sonnet 5.5 "disabled" da error 400
    return {"type": "disabled"}  # Haiku 5.5


def build_claude_request(model, effort, thinking_on, system, messages) -> dict:
    request = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "output_config": {"effort": effort},
        "thinking": config_thinking(model, thinking_on),
    }
    if system.strip():
        request["system"] = system.strip()
    return request


def build_ollama_request(model, thinking_on, system, messages) -> dict:
    # En Ollama el system prompt va como primer mensaje con role "system"
    full = ([{"role": "system", "content": system.strip()}] if system.strip() else []) + messages
    request = {"model": model, "messages": full}
    if thinking_on:
        request["think"] = True
    return request


# ---------------------------------------------------------------------------
# Llamadas a cada proveedor. Regresan (texto, razonamiento)
# ---------------------------------------------------------------------------
def call_claude(request: dict) -> tuple[str, str]:
    client = anthropic.Anthropic()
    response = client.messages.create(**request)

    if response.stop_reason == "refusal":
        return "Claude no quiso responder esta petición.", ""

    text = "".join(b.text for b in response.content if b.type == "text")
    thinking = "\n\n".join(b.thinking for b in response.content if b.type == "thinking" and b.thinking)
    if response.stop_reason == "max_tokens":
        text += "\n\n_(La respuesta se cortó: llegó al límite de max_tokens.)_"
    st.session_state.last_usage = (response.usage.input_tokens, response.usage.output_tokens)
    return text, thinking


def call_ollama(request: dict) -> tuple[str, str]:
    response = ollama.chat(**request)
    return response.message.content or "", response.message.thinking or ""


@st.cache_data(ttl=60)
def ollama_models() -> dict[str, bool]:
    """Modelos de chat descargados -> ¿soporta thinking?"""
    models = {}
    for m in ollama.list().models:
        caps = ollama.show(m.model).capabilities or []
        if "completion" in caps:  # descarta los de embeddings
            models[m.model] = "thinking" in caps
    return models


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")
    provider = st.radio(
        "Proveedor",
        ["Claude (Anthropic)", "Ollama (local)", "Modo de prueba"],
        help="Usa Ollama o el modo de prueba mientras no tengas una API key válida.",
    )

    if provider == "Ollama (local)":
        try:
            models = ollama_models()
        except Exception:
            models = {}
            st.error("No pude conectar con Ollama. ¿Está abierto?")
        if not models:
            st.stop()
        model = st.selectbox("Modelo", list(models))
        supports_thinking = models[model]
        thinking_on = st.toggle("Thinking", value=False, disabled=not supports_thinking)
        if not supports_thinking:
            st.caption(f"`{model}` no soporta thinking.")
        effort = None
        st.caption("Effort es un parámetro de Claude; Ollama lo ignora.")
    else:
        label = st.selectbox("Modelo", list(CLAUDE_MODELS))
        model = CLAUDE_MODELS[label]
        is_opus = "opus" in model
        thinking_on = st.toggle("Thinking", value=True, disabled=is_opus) or is_opus
        if is_opus:
            st.caption("En Opus 5.5 el thinking no se puede apagar; usa effort bajo para que piense menos.")
        options = EFFORTS if thinking_on else EFFORTS_SIN_THINKING
        effort = st.select_slider("Effort", options=options, value="medium")
        if not thinking_on:
            st.caption("Sin thinking, effort solo llega hasta `high`.")

    system = st.text_area("System prompt", "Eres un asistente útil. Responde en español.", height=120)

    if st.button("Borrar conversación"):
        st.session_state.messages = []
        st.session_state.pop("last_usage", None)
        st.rerun()

    if usage := st.session_state.get("last_usage"):
        st.caption(f"Último mensaje: {usage[0]} tokens de entrada, {usage[1]} de salida")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
st.title("💬 Chat con Claude")
st.caption(f"{provider} · `{model}`" + (f" · effort `{effort}`" if effort else "") + f" · thinking {'on' if thinking_on else 'off'}")

if "messages" not in st.session_state:
    st.session_state.messages = []  # el modelo no recuerda nada: aquí guardamos el historial

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("thinking"):
            with st.expander("Razonamiento"):
                st.write(m["thinking"])
        st.markdown(m["content"])

if prompt := st.chat_input("Escribe tu mensaje"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Solo role + content van al modelo (sin el razonamiento guardado para la UI)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    if provider == "Ollama (local)":
        request = build_ollama_request(model, thinking_on, system, history)
    else:
        request = build_claude_request(model, effort, thinking_on, system, history)

    with st.chat_message("assistant"):
        with st.expander("Petición enviada"):
            st.code(json.dumps(request, indent=2, ensure_ascii=False), language="json")

        text, thinking = "", ""
        try:
            with st.spinner("Pensando..."):
                if provider == "Claude (Anthropic)":
                    text, thinking = call_claude(request)
                elif provider == "Ollama (local)":
                    text, thinking = call_ollama(request)
                else:
                    text = "_Modo de prueba: no se llamó a ningún modelo. Revisa arriba la petición que se habría mandado._"
        except anthropic.AuthenticationError:
            st.error("La API key no es válida (401). Revisa el .env o cambia a Ollama / Modo de prueba.")
        except anthropic.BadRequestError as e:
            st.error(f"Claude rechazó la petición (400): {e.message}")
        except anthropic.RateLimitError:
            st.error("Demasiadas peticiones (429). Espera un momento y vuelve a intentar.")
        except anthropic.APIStatusError as e:
            st.error(f"Error de la API ({e.status_code}): {e.message}")
        except anthropic.APIConnectionError:
            st.error("No pude conectar con la API de Anthropic. Revisa tu internet.")
        except ollama.ResponseError as e:
            st.error(f"Ollama respondió con error: {e.error}")
        except ConnectionError:
            st.error("No pude conectar con Ollama. ¿Está abierto?")

        if thinking:
            with st.expander("Razonamiento"):
                st.write(thinking)
        if text:
            st.markdown(text)

    if text:
        st.session_state.messages.append({"role": "assistant", "content": text, "thinking": thinking})
    else:
        st.session_state.messages.pop()  # si falló, quita la pregunta para no dejar el historial cojo
