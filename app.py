import os
import requests
import streamlit as st

# ---- LLM backend helpers ----

def _openai_available():
    return bool(os.getenv("OPENAI_API_KEY"))

def _ollama_available(base_url="http://localhost:11434"):
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_openai(messages, model="gpt-4.1-mini"):
    client = _get_openai_client()
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

def ask_ollama(messages, model="llama3.1:8b", base_url="http://localhost:11434"):
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

def ask_llm(messages):
    backend = st.session_state.get("llm_backend", "auto")
    if backend == "OpenAI" or (backend == "auto" and _openai_available()):
        return ask_openai(messages)
    elif backend == "Ollama (local)" or (backend == "auto" and _ollama_available()):
        ollama_model = st.session_state.get("ollama_model", "llama3.1:8b")
        return ask_ollama(messages, model=ollama_model)
    else:
        return "⚠️ No LLM backend available. Set OPENAI_API_KEY or start Ollama (`ollama serve`)."

import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape

st.set_page_config(layout="wide")

# ---------- SIDEBAR: LLM BACKEND ----------
with st.sidebar:
    st.header("LLM Settings")
    _has_openai = _openai_available()
    _has_ollama = _ollama_available()

    _available = []
    if _has_openai:
        _available.append("OpenAI")
    if _has_ollama:
        _available.append("Ollama (local)")
    _available.append("auto")

    _default_backend = "OpenAI" if _has_openai else ("Ollama (local)" if _has_ollama else "auto")
    st.session_state["llm_backend"] = st.selectbox(
        "Backend",
        _available,
        index=_available.index(_default_backend),
        help="OpenAI requires OPENAI_API_KEY. Ollama requires `ollama serve` running locally."
    )

    if st.session_state["llm_backend"] == "Ollama (local)":
        st.session_state["ollama_model"] = st.text_input(
            "Ollama model", value="llama3.1:8b"
        )

    if not _has_openai and not _has_ollama:
        st.warning("No LLM backend detected. Set OPENAI_API_KEY or run `ollama serve`.")
    else:
        _status = []
        if _has_openai:
            _status.append("✅ OpenAI")
        else:
            _status.append("❌ OpenAI (no key)")
        if _has_ollama:
            _status.append("✅ Ollama")
        else:
            _status.append("❌ Ollama (not running)")
        st.caption("  \n".join(_status))

# ---------- CSS ----------
st.markdown(
    """
    <style>

    }
    .panel-title {
        font-size: 0.5rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
        color: #f9fafb;
    }

    .map-panel iframe {
    width: 100% !important;
    min-height: 540px;
    border-radius: 10px;
    }

    div[data-testid="stChatInput"] {
        max-width: 25%;
        margin-left: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------- CHAT STATE ----------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "bot", "content": "Ask me anything about urban design..."}
    ]

# ---------- DATA LOADING ----------
@st.cache_data(show_spinner=False)
def load_data():
    gdf = gpd.read_file("data/OneNeighborhood.shp")
    return gdf.to_crs(epsg=4326)

gdf = load_data()

# ---------- MAP PREP ----------
# centers
center_geom = gdf.to_crs(epsg=3857).union_all().centroid
center = (
    gpd.GeoSeries([center_geom], crs=3857)
    .to_crs(epsg=4326)
    .iloc[0]
)

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=15,
    tiles="cartodbpositron"
)

folium.GeoJson(
    gdf,
    style_function=lambda _: {
        "fillColor": "#3186cc",
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.3,
    },
).add_to(m)

Draw(
    draw_options={
        "polygon": True,
        "rectangle": True,
        "polyline": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
    },
    edit_options={"edit": False}
).add_to(m)

st.title("Building Selection UI")

# ---------- COLUMNS ----------
col1, col2, col3 = st.columns([1, 2, 1])

# ---------- LEFT PANEL ----------
with col1:
    st.subheader("Chat")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">CHATBOT</div>', unsafe_allow_html=True)

    # showing previous messages
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='text-align:right; margin-bottom:6px;'>"
                f"<span style='background:#2563eb; color:white; padding:6px 10px; border-radius:10px;'>"
                f"{msg['content']}</span></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left; margin-bottom:6px;'>"
                f"<span style='background:#374151; color:white; padding:6px 10px; border-radius:10px;'>"
                f"{msg['content']}</span></div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # user message
    user_input = st.text_input(
        "Type your message",
        key="chat_input",
        label_visibility="collapsed"
    )

    # send button
    if st.button("Send"):
        if user_input.strip() != "":
            # saving messages
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )

            # build message history for LLM
            llm_messages = [
                {"role": "system", "content": "You are an urban planning assistant."}
            ]

            for msg in st.session_state.chat_messages:
                llm_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            # get reply from active backend
            bot_reply = ask_llm(llm_messages)

            # saving the answer
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": bot_reply}
            )

            # clean input
            st.session_state.chat_input = ""

    st.markdown('</div>', unsafe_allow_html=True)



# ---------- MAP ----------
with col2:
    st.subheader("Selection On Map")
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    map_output = st_folium(
        m,
        height=550,
        width='stretch',
        key="main_map"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- SELECTION LOGIC ----------
selected_gdf = gdf.copy()

# Filter by drawn geometry
if map_output and map_output.get("last_active_drawing"):
    drawn_geom = shape(map_output["last_active_drawing"]["geometry"])
    selected_gdf = selected_gdf[selected_gdf.intersects(drawn_geom)]

# ---------- RIGHT PANEL ----------
with col3:
    st.subheader("Listed documents")
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if not selected_gdf.empty:
        st.success(f"✅ {len(selected_gdf)} building selected.")
        st.dataframe(
            selected_gdf.drop(columns="geometry"),
            width='stretch'
        )
    else:
        st.info("Use the tools to select a site on the map")

    st.markdown('</div>', unsafe_allow_html=True)

    

