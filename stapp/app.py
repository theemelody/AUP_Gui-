import os
import streamlit as st
import requests
import geopandas as gpd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape
import zipfile
import tempfile
from io import BytesIO

st.set_page_config(layout="wide", page_title="Urban Design Assistant")

# ---------- CSS ----------
st.markdown(
    """
    <style>
    .panel-title {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
        color: #f9fafb;
    }

    /* Chat input width */
    div[data-testid="stChatInput"] {
        width: 100% !important;
    }

    /* Chat textarea height and border */
    div[data-testid="stChatInput"] textarea {
        height: 50px !important;
        border-radius: 10px !important;
        border: 1px solid #444 !important;
    }

    /* Normal button */
    div[data-testid="stButton"] button {
        height: 50px !important;
        border-radius: 10px !important;
        border: 1px solid #444 !important;
        font-weight: 500;
    }

    /* File uploader button */
    div[data-testid="stFileUploader"] button {
        height: 50px !important;
        border-radius: 10px !important;
        border: 1px solid #444 !important;
        font-weight: 500;
        width: 100% !important;
    }

    /* Hover efekti */
    button:hover {
        border: 1px solid #888 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_response(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]
    except:
        return "The connection to Ollama could not be established. Please ensure the model is working."

# ---------- DATA LOADING ----------
@st.cache_data(show_spinner=False)
def load_data():
    gdf = gpd.read_file("data/OneNeighborhood.shp")
    return gdf.to_crs(epsg=4326)

try:
    gdf = load_data()
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

# ---------- MAP PREP ----------
center_geom = gdf.to_crs(epsg=3857).union_all().centroid
center = gpd.GeoSeries([center_geom], crs=3857).to_crs(epsg=4326).iloc[0]

m = folium.Map(location=[center.y, center.x], zoom_start=15, tiles="cartodbpositron")

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
    draw_options={"polygon": True, "rectangle": True, "polyline": False, "circle": False, "marker": False},
    edit_options={"edit": False}
).add_to(m)

st.title("Building Selection UI")

# ---------- COLUMNS ----------
col1, col2, col3 = st.columns([1.2, 2, 1])

# ---------- LEFT PANEL (CHAT) ----------
with col1:
    st.subheader("Urban AI Chat")
    
    chat_container = st.container()
    with chat_container:
        chat_html = """<div style="height:500px; overflow-y:auto; border:1px solid #444; padding:15px; border-radius:10px; background-color:#111;">"""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"<p style='color:#4ade80;'><b>You:</b> {msg['content']}</p>"
            else:
                chat_html += f"<p style='color:#60a5fa;'><b>AI:</b> {msg['content']}</p>"
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    user_input = st.chat_input("Ask me anything about urban design...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        bot_reply = generate_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()

# ---------- MIDDLE PANEL (MAP) ----------
with col2:
    st.subheader("Selection On Map")
    map_output = st_folium(m, height=500, width=None, key="main_map")

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        run_simulation = st.button("Run Simulation", use_container_width=True)

    with btn_col2:
        uploaded_file = st.file_uploader( 
            "Upload Document",
            label_visibility="collapsed"
        )

    if run_simulation:
        st.success("Simulation started...")

# ---------- SELECTION LOGIC ----------
selected_gdf = gdf.copy()
if map_output and map_output.get("last_active_drawing"):
    drawn_geom = shape(map_output["last_active_drawing"]["geometry"])
    # CRS mapping for intersection control
    selected_gdf = selected_gdf[selected_gdf.intersects(drawn_geom)]

# ---------- RIGHT PANEL (DATA) ----------
with col3:
    st.subheader("Listed Buildings")

    if not selected_gdf.empty and len(selected_gdf) != len(gdf):
        st.success(f"✅ {len(selected_gdf)} buildings selected.")
        st.dataframe(selected_gdf.drop(columns="geometry"), use_container_width=True)
        
        # --- SHP Export ---
        tmpdir = tempfile.mkdtemp()
        shp_path = os.path.join(tmpdir, "selected_buildings.shp")
        selected_gdf.to_file(shp_path)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            for file in os.listdir(tmpdir):
                z.write(os.path.join(tmpdir, file), arcname=file)
        zip_buffer.seek(0)

        st.download_button(
            label="Download as Shapefile",
            data=zip_buffer,
            file_name="selected_buildings.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("Use the draw tools on the map to select buildings.")