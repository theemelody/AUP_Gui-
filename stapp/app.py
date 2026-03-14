import os
import subprocess
import requests
import glob
from datetime import datetime
from pathlib import Path
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
import pandas as pd
import numpy as np
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape
import charts_plotly as cp


_APP_DIR = Path(__file__).resolve().parent
_AUP_ROOT = _APP_DIR.parent
_PROJECT_ROOT = _AUP_ROOT.parent
_SCENARIOS_DIR = _PROJECT_ROOT / "scenarios"
_DEFAULT_SCENARIO_PATH = _SCENARIOS_DIR / "output-scenario"
_DEFAULT_CEA_DB_PATH = _PROJECT_ROOT / "CityEnergyAnalyst" / "cea" / "databases" / "DE"


def prepare_scenario(scenario_path: Path, zone_gdf):
    """Create CEA scenario folder structure and write zone/site shapefiles."""
    for subdir in ["building-geometry", "building-properties", "topography", "weather"]:
        (scenario_path / "inputs" / subdir).mkdir(parents=True, exist_ok=True)
    (scenario_path / "outputs").mkdir(parents=True, exist_ok=True)

    geom_dir = scenario_path / "inputs" / "building-geometry"
    zone_gdf.to_file(geom_dir / "zone.shp")

    site_gdf = gpd.GeoDataFrame(geometry=[zone_gdf.union_all()], crs=zone_gdf.crs)
    site_gdf.to_file(geom_dir / "site.shp")


_CEA_BIN = os.environ.get("CEA_BIN", "/home/salva/micromamba/envs/cea/bin/cea")


def build_cea_steps(scenario_path: str, databases_path: str) -> list:
    s = str(scenario_path)
    return [
        ("database-helper",     [_CEA_BIN, "database-helper",     "--scenario", s, "--databases-path", databases_path]),
        ("zone-helper",         [_CEA_BIN, "zone-helper",         "--scenario", s]),
        ("weather-helper",      [_CEA_BIN, "weather-helper",      "--scenario", s]),
        ("surroundings-helper", [_CEA_BIN, "surroundings-helper", "--scenario", s]),
        ("terrain-helper",      [_CEA_BIN, "terrain-helper",      "--scenario", s]),
        ("archetypes-mapper",   [_CEA_BIN, "archetypes-mapper",   "--scenario", s]),
        ("radiation",           [_CEA_BIN, "radiation",           "--scenario", s]),
        ("occupancy",           [_CEA_BIN, "occupancy",           "--scenario", s]),
        ("demand",              [_CEA_BIN, "demand",              "--scenario", s]),
    ]


# ---------- DEMAND DATA LOADER ------------------------------------------------
_DEFAULT_EF = {
    'NG': 0.202, 'COAL': 0.341, 'OIL': 0.266, 'WOOD': 0.017,
    'DH': 0.070, 'GRID': 0.366, 'SOLAR': 0.000, 'PV': 0.000,
}

@st.cache_data(show_spinner=False)
def load_demand(scenario_dir: str):
    """Load annual + hourly demand CSVs and solar-radiation files."""
    demand_dir = Path(scenario_dir) / "outputs" / "data" / "demand"
    rad_dir    = Path(scenario_dir) / "outputs" / "data" / "solar-radiation"

    annual_csv  = demand_dir / "Total_demand.csv"
    hourly_csv  = demand_dir / "Total_demand_hourly.csv"

    if not annual_csv.exists():
        return None, None

    df  = pd.read_csv(annual_csv).set_index("name")
    dfh = pd.read_csv(hourly_csv, parse_dates=["date"]).set_index("date") if hourly_csv.exists() else None

    # theta_o_C in the Total file is the SUM across all buildings; convert to mean
    if dfh is not None and "theta_o_C" in dfh.columns:
        n_bldgs = len(df)
        if n_bldgs > 1:
            dfh["theta_o_C"] = dfh["theta_o_C"] / n_bldgs

    # Attach solar roof irradiation
    rad_data = {}
    for fp in sorted(glob.glob(str(rad_dir / "*_radiation.csv"))):
        bname = Path(fp).stem.replace("_radiation", "")
        r = pd.read_csv(fp, parse_dates=["Date"])
        rad_data[bname] = r["roofs_top_kW"].sum() / 1e3
    df["solar_roof_MWh"] = pd.Series(rad_data)
    return df, dfh


def _district_kpis(df, EF):
    total_heat  = df["QH_sys_MWhyr"].sum()
    total_cool  = df["QC_sys_MWhyr"].sum()
    total_grid  = df["GRID_MWhyr"].sum()
    total_gfa   = df["GFA_m2"].sum()
    fossil_mwh  = df[["NG_hs_MWhyr","NG_ww_MWhyr",
                       "COAL_hs_MWhyr","COAL_ww_MWhyr",
                       "OIL_hs_MWhyr","OIL_ww_MWhyr"]].sum(axis=1).sum()
    total_final = (
        total_grid
        + df[["NG_hs_MWhyr","NG_ww_MWhyr","COAL_hs_MWhyr","COAL_ww_MWhyr",
              "OIL_hs_MWhyr","OIL_ww_MWhyr","WOOD_hs_MWhyr","WOOD_ww_MWhyr",
              "DH_hs_MWhyr","DH_ww_MWhyr","SOLAR_hs_MWhyr","SOLAR_ww_MWhyr"]]
             .sum(axis=1).sum()
    )
    co2_fossil  = (
        (df["NG_hs_MWhyr"] + df["NG_ww_MWhyr"]).sum()   * EF["NG"]
        + (df["COAL_hs_MWhyr"] + df["COAL_ww_MWhyr"]).sum() * EF["COAL"]
        + (df["OIL_hs_MWhyr"] + df["OIL_ww_MWhyr"]).sum()  * EF["OIL"]
    )
    total_co2   = (
        total_grid * EF["GRID"]
        + co2_fossil
        + (df["WOOD_hs_MWhyr"] + df["WOOD_ww_MWhyr"]).sum() * EF["WOOD"]
        + (df["DH_hs_MWhyr"]   + df["DH_ww_MWhyr"]).sum()   * EF["DH"]
    )
    eui_now     = total_final / total_gfa * 1000 if total_gfa > 0 else 0
    fossil_pct  = fossil_mwh / total_final * 100 if total_final > 0 else 0
    return dict(
        total_heat=total_heat, total_cool=total_cool, total_grid=total_grid,
        total_gfa=total_gfa, total_co2=total_co2, eui_now=eui_now,
        fossil_mwh=fossil_mwh, co2_fossil=co2_fossil, fossil_pct=fossil_pct,
    )


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

    st.divider()
    st.header("Simulation Settings")
    # Initialise session state keys
    if "scenario_name_confirmed" not in st.session_state:
        st.session_state.scenario_name_confirmed = False
    if "sim_scenario_path" not in st.session_state:
        st.session_state.sim_scenario_path = ""
    st.session_state.cea_db_path = st.text_input(
        "CEA databases path",
        value=st.session_state.get(
            "cea_db_path",
            str(_DEFAULT_CEA_DB_PATH)
        )
    )

    st.divider()
    st.subheader("Saved Scenarios")

    _scenarios_dir = _SCENARIOS_DIR
    _saved = sorted(
        [d for d in _scenarios_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    ) if _scenarios_dir.exists() else []

    if not _saved:
        st.caption("No saved scenarios yet.")
    else:
        _active = st.session_state.get("sim_scenario_path", "")
        for _sc in _saved:
            _has_data = (_sc / "outputs" / "data" / "demand" / "Total_demand.csv").exists()
            _label = f"{'✅' if _has_data else '⏳'} {_sc.name}"
            _is_active = str(_sc) == _active
            if st.button(
                _label,
                width="stretch",
                key=f"load_sc_{_sc.name}",
                type="primary" if _is_active else "secondary",
                disabled=not _has_data,
            ):
                st.session_state.sim_scenario_path = str(_sc)
                st.session_state.scenario_name_confirmed = True
                st.session_state["_scenario_raw_name"] = _sc.name.removesuffix("-scenario")
                st.session_state["_scenario_overwrite"] = False
                load_demand.clear()
                st.rerun()

    st.divider()
    st.subheader("Run Simulation")

    # ── Scenario name ────────────────────────────────────────────────────────
    _sb_sn_col, _sb_sv_col = st.columns([4, 1])
    with _sb_sn_col:
        _scenario_input = st.text_input(
            "Scenario name",
            value=st.session_state.get("_scenario_raw_name", ""),
            placeholder="e.g. munich-baseline",
            label_visibility="collapsed",
        )
    with _sb_sv_col:
        _save_pressed = st.button("Save", width="stretch", key="sb_save")

    if _save_pressed and _scenario_input.strip():
        _raw = _scenario_input.strip()
        _folder = f"{_raw}-scenario"
        _full_path = _SCENARIOS_DIR / _folder
        st.session_state["_scenario_raw_name"] = _raw
        st.session_state.sim_scenario_path = str(_full_path)
        st.session_state.scenario_name_confirmed = True
        st.session_state["_scenario_overwrite"] = _full_path.exists()
    elif _save_pressed:
        st.warning("Please enter a scenario name before saving.")

    _sb_confirmed = st.session_state.get("scenario_name_confirmed", False)
    _sb_overwrite = st.session_state.get("_scenario_overwrite", False)
    _sb_sel = st.session_state.get("_selected_gdf", gpd.GeoDataFrame())

    if _sb_confirmed:
        st.caption(f"📁 `{st.session_state.sim_scenario_path}`")
    if _sb_overwrite:
        st.warning("⚠️ A scenario with this name already exists. Continuing will overwrite it.")

    if st.button("🚀 Run Simulation", width="stretch",
                 disabled=(_sb_sel.empty or not _sb_confirmed), key="sb_run"):
        _run_path = Path(st.session_state.get("sim_scenario_path", str(_DEFAULT_SCENARIO_PATH)))
        _db_path  = st.session_state.get("cea_db_path", "")
        _run_gdf  = st.session_state.get("_selected_gdf", gpd.GeoDataFrame())

        with st.status("Running CEA simulation...", expanded=True) as _sim_status:
            st.write("📁 Preparing scenario structure...")
            try:
                _run_path.mkdir(parents=True, exist_ok=True)
                prepare_scenario(_run_path, _run_gdf)
                st.write(f"✅ Scenario ready: `{_run_path}`")
            except Exception as _e:
                _sim_status.update(label="❌ Scenario preparation failed", state="error")
                st.error(str(_e))
                st.stop()

            _failed_at = None
            for _label, _cmd in build_cea_steps(_run_path, _db_path):
                st.write(f"⏳ Running `{_label}`...")
                try:
                    _result = subprocess.run(
                        _cmd, capture_output=True, text=True, timeout=1800
                    )
                except subprocess.TimeoutExpired:
                    st.error(f"`{_label}` timed out after 30 minutes.")
                    _failed_at = _label
                    break
                if _result.returncode != 0:
                    st.error(_result.stderr[-1000:] or _result.stdout[-1000:])
                    _failed_at = _label
                    break
                st.write(f"✅ `{_label}` done")

            if _failed_at:
                _sim_status.update(label=f"❌ Failed at {_failed_at}", state="error")
            else:
                _sim_status.update(label="✅ Simulation complete!", state="complete")
                st.success(f"Results saved to `{_run_path}/outputs/data/demand/`")
                st.session_state["_scenario_overwrite"] = False
                load_demand.clear()

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

    /* ── 3-second hover tooltip ─────────────────────────────────────── */
    .tip-wrap {
        position: relative;
        display: inline-block;
        cursor: help;
        margin: 2px 0 6px;
    }
    .tip-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px; height: 22px;
        border-radius: 50%;
        background: #4472C4;
        color: #fff;
        font-size: 13px;
        font-weight: bold;
        user-select: none;
    }
    .tip-box {
        visibility: hidden;
        opacity: 0;
        width: 360px;
        background: #1e1e2e;
        color: #dde;
        font-size: 12.5px;
        line-height: 1.6;
        border-radius: 8px;
        padding: 12px 16px;
        position: absolute;
        z-index: 9999;
        left: 28px; top: -4px;
        border: 1px solid #555;
        box-shadow: 0 6px 24px rgba(0,0,0,.55);
        transition: visibility 0s linear, opacity 0.25s ease;
    }
    .tip-wrap:hover .tip-box {
        visibility: visible;
        opacity: 1;
        transition: visibility 0s linear 3s, opacity 0.25s ease 3s;
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
def _resolve_zone_shp() -> Path:
    candidates = [
        _PROJECT_ROOT / "munich-scenario" / "inputs" / "building-geometry" / "zone.shp",
        _PROJECT_ROOT / "scenarios" / "munich-commercial-scenario" / "inputs" / "building-geometry" / "zone.shp",
        _PROJECT_ROOT / "reference-case-open" / "baseline" / "inputs" / "building-geometry" / "zone.shp",
        _AUP_ROOT / "data" / "OneNeighborhood.shp",
    ]

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"No scenario geometry file found. Checked:\n{checked}")

@st.cache_data(show_spinner=False)
def load_data():
    gdf = gpd.read_file(str(_resolve_zone_shp()))
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

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message",
            key="chat_input",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
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

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": bot_reply}
        )
        st.rerun()

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

# Persist so sidebar can access it after rerender
st.session_state["_selected_gdf"] = selected_gdf

# ---------- RIGHT PANEL ----------
with col3:
    st.subheader("Listed documents")
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    _sel = st.session_state.get("_selected_gdf", selected_gdf)
    if not _sel.empty:
        st.success(f"✅ {len(_sel)} building(s) selected.")
        st.dataframe(
            _sel.drop(columns="geometry"),
            width='stretch'
        )
    else:
        st.info("Use the tools to select a site on the map")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# CHARTS TAB
# ============================================================

def _tip(html_text: str) -> str:
    """Render an info icon that shows a tooltip after hovering for 3 seconds."""
    return (
        f'<div class="tip-wrap">'
        f'<span class="tip-icon">ⓘ</span>'
        f'<div class="tip-box">{html_text}</div>'
        f'</div>'
    )

st.divider()
st.subheader("📊 SECAP Charts")

_scenario_dir = st.session_state.get("sim_scenario_path", "")
df_dem, dfh_dem = load_demand(_scenario_dir) if _scenario_dir else (None, None)

if df_dem is None:
    st.info("Run a simulation first — charts will appear here once demand data is available.")
else:
    kpis = _district_kpis(df_dem, _DEFAULT_EF)

    # ── KPI metric row ──────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Buildings",     f"{len(df_dem)}")
    k2.metric("GFA",           f"{kpis['total_gfa']:,.0f} m²")
    k2.markdown(_tip(
        "<b>GFA — Gross Floor Area</b><br><br>"
        "Total built floor space of all district buildings in m². "
        "Used as the denominator for energy intensity (<b>EUI</b>) and per-area CO₂ metrics."
    ), unsafe_allow_html=True)
    k3.metric("Total CO₂",     f"{kpis['total_co2']:,.0f} tCO₂/yr")
    k3.markdown(_tip(
        "<b>Total CO₂ — Annual District Emissions</b><br><br>"
        "Sum of CO₂-equivalent emissions from all energy carriers (electricity, heating, cooling, DHW). "
        "Computed as: Σ (Energy<sub>carrier</sub> × EF<sub>carrier</sub>), "
        "where <b>EF</b> = emission factor (tCO₂/MWh). "
        "Unit: tonnes of CO₂ per year (tCO₂/yr). "
        "This is the <b>BEI</b> (Baseline Emission Inventory) figure for <b>SECAP §3</b>."
    ), unsafe_allow_html=True)
    k4.metric("EUI",           f"{kpis['eui_now']:,.0f} kWh/m²/yr")
    k4.markdown(_tip(
        "<b>EUI — Energy Use Intensity</b><br><br>"
        "Total site energy demand divided by <b>GFA</b>. "
        "Measures the energy efficiency of the district as a whole. "
        "Lower values indicate a more efficient building stock. "
        "Unit: kWh per square metre per year (kWh/m²/yr). "
        "Reference benchmarks: passive house ≈ 15, well-retrofitted ≈ 60, "
        "typical existing stock ≈ 120–200 kWh/m²/yr."
    ), unsafe_allow_html=True)
    k5.metric("Fossil share",  f"{kpis['fossil_pct']:,.1f}%")
    k5.markdown(_tip(
        "<b>Fossil Share — % of Demand from Fossil Fuels</b><br><br>"
        "Percentage of total annual energy demand supplied by fossil carriers "
        "(natural gas <b>NG</b>, coal, oil). "
        "Renewables (<b>PV</b>, solar thermal), district heating (<b>DH</b>), "
        "and grid electricity are excluded from the fossil count. "
        "Reducing this share is a core <b>SECAP</b> mitigation lever."
    ), unsafe_allow_html=True)
    k6.metric("2030 CO₂ target", f"{kpis['total_co2']*0.6:,.0f} tCO₂/yr")
    k6.markdown(_tip(
        "<b>2030 CO₂ Target — SECAP Mitigation Goal</b><br><br>"
        "The minimum emission reduction required by the EU <b>SECAP</b> framework: "
        "a <b>40% cut</b> from the baseline by 2030 "
        "(updated Covenant of Mayors target). "
        "Displayed value = Total CO₂ × 0.6. "
        "Compare against the gauges in Chart B to track progress."
    ), unsafe_allow_html=True)

    st.divider()

    # ── BEI row: Charts 1, A, C ─────────────────────────────────────────────
    st.markdown("**§3 Baseline Emission Inventory (BEI)**")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(cp.chart_energy_mix(df_dem, _DEFAULT_EF), width="stretch")
        st.markdown(_tip(
            "<b>Chart 1 — Energy Mix &amp; CO₂ Baseline</b> &nbsp;<i>SECAP §3 BEI</i><br><br>"
            "<b>SECAP</b> = Sustainable Energy &amp; Climate Action Plan (Covenant of Mayors framework)<br>"
            "<b>BEI</b> = Baseline Emission Inventory — the starting-point audit of all energy use and CO₂ from buildings<br>"
            "<b>NG</b> = Natural Gas &nbsp;·&nbsp; <b>DH</b> = District Heating &nbsp;·&nbsp; <b>PV</b> = Photovoltaic solar<br><br>"
            "Shows district total final energy split by fuel source, plus CO₂ per carrier via emission factors.<br><br>"
            "<b>⚠️ Scope note:</b> This chart covers <b>buildings only</b> (Scope 1 + Scope 2). "
            "A full municipal SECAP also accounts for:<br>"
            "&nbsp;• <b>Transport</b> — private cars, buses, freight (often the largest sector)<br>"
            "&nbsp;• <b>Street lighting &amp; public facilities</b> — municipal electricity<br>"
            "&nbsp;• <b>Waste &amp; wastewater</b> — methane from landfill / treatment<br>"
            "&nbsp;• <b>Industry</b> — if within municipal boundary<br><br>"
            "Key drivers of building emissions: <b>grid electricity EF</b> (dirtier grid = more CO₂ per kWh), "
            "<b>heating system fuel</b> (gas boiler vs. heat pump), "
            "<b>building envelope</b> (insulation quality determines total demand), "
            "and <b>occupancy &amp; behaviour</b> (actual vs. calculated use)."
        ), unsafe_allow_html=True)
        with st.expander("ℹ️ About this chart"):
            st.markdown(
                "**Chart 1 — Energy Mix & CO₂ Baseline** · *SECAP §3 BEI*\n\n"
                "Shows district total final energy broken down by energy carrier "
                "(grid electricity, natural gas, district heating, solar thermal, oil, coal, wood/biomass). "
                "The red ◆ markers show estimated CO₂ emissions per carrier using standard emission factors "
                "(Grid: 0.366 tCO₂/MWh; NG: 0.202; DH: 0.070; PV/Solar: 0.000). "
                "This is the primary input for the BEI narrative and identifies which fuels drive the most emissions.\n\n"
                "**Scope of this chart:** Buildings only (Scope 1 direct combustion + Scope 2 purchased electricity). "
                "A complete SECAP BEI also includes transport, street lighting, public facilities, and waste.\n\n"
                "**Key emission drivers:** grid electricity emission factor, heating fuel type, "
                "building envelope quality (insulation → total demand), and occupant behaviour."
            )
    with r1c2:
        st.plotly_chart(cp.chart_end_use_breakdown(df_dem), width="stretch")
        st.markdown(_tip(
            "<b>Chart A — End-Use Breakdown</b> &nbsp;<i>SECAP §3 BEI</i><br><br>"
            "<b>BEI</b> = Baseline Emission Inventory<br>"
            "<b>DHW</b> = Domestic Hot Water (showers, taps, washing, etc.)<br>"
            "<b>End-use</b> = what energy is actually used for inside the building<br><br>"
            "Splits total delivered energy between space heating, DHW, cooling, appliances, and lighting — the key question for targeting retrofits."
        ), unsafe_allow_html=True)
        with st.expander("ℹ️ About this chart"):
            st.markdown(
                "**Chart A — End-Use Breakdown** · *SECAP §3 BEI*\n\n"
                "Splits delivered energy by end-use: space heating, domestic hot water (DHW), "
                "cooling, appliances, and lighting. Answers the question *'where is the energy actually going?'* — "
                "the most important context for targeting mitigation measures at the uses that matter most."
            )

    fig_sector = cp.chart_sector_mix(df_dem)
    if fig_sector:
        st.plotly_chart(fig_sector, width="stretch")
        st.markdown(_tip(
            "<b>Chart C — Building Sector Mix</b> &nbsp;<i>SECAP §3 BEI</i><br><br>"
            "<b>GFA</b> = Gross Floor Area — total usable floor space measured in m²<br>"
            "<b>EUI</b> = Energy Use Intensity — how much energy a building uses per m² per year (kWh/m²/yr); lower is better<br>"
            "<b>BEI</b> = Baseline Emission Inventory<br><br>"
            "Groups buildings by use type (residential, office, retail, municipal) and compares GFA share, energy share, and sector EUI. Tertiary buildings typically have 2–3× higher EUI than residential."
        ), unsafe_allow_html=True)
        with st.expander("ℹ️ About this chart"):
            st.markdown(
                "**Chart C — Building Sector Mix** · *SECAP §3 BEI*\n\n"
                "Groups buildings by primary use type (residential, office, retail, municipal, etc.) "
                "and shows how GFA and energy consumption are distributed across sectors, plus EUI per sector. "
                "Tertiary buildings typically have 2–3× higher EUI than residential and are priority targets. "
                "SECAP §3 requires a sector breakdown to identify key consumption categories."
            )

    st.divider()

    # ── Commitments: Chart B ────────────────────────────────────────────────
    st.markdown("**§2.2 Commitments — 2030 Target**")
    st.plotly_chart(
        cp.chart_target_gauge(kpis["total_co2"], kpis["eui_now"]),
        width="stretch",
    )
    st.markdown(_tip(
        "<b>Chart B — 2030 Target Gauge</b> &nbsp;<i>SECAP §2.2 Commitments</i><br><br>"
        "<b>SECAP</b> = Sustainable Energy &amp; Climate Action Plan — the climate plan municipalities submit to the Covenant of Mayors<br>"
        "<b>EUI</b> = Energy Use Intensity (kWh/m²/yr)<br>"
        "<b>−40 % target</b> = Covenant of Mayors legal commitment: cut CO₂ emissions by at least 40 % from the baseline year by 2030<br><br>"
        "Gauge showing how far current CO₂ and EUI sit above (red zone) or below (green zone) the 2030 legal target."
    ), unsafe_allow_html=True)
    with st.expander("ℹ️ About this chart"):
        st.markdown(
            "**Chart B — Emissions vs 2030 Target Gauge** · *SECAP §2.2 Commitments*\n\n"
            "A visual gauge showing current baseline CO₂ (tCO₂/yr) and district EUI (kWh/m²/yr) "
            "against the legally binding 2030 target ceiling — **60% of baseline (−40% reduction)** as required "
            "by the Covenant of Mayors. The green zone is the safe region; the red zone shows how far above target "
            "the district currently sits. This is the first chart SECAP reviewers look for."
        )

    st.divider()

    # ── EUI + Solar: Charts 5, 6 ────────────────────────────────────────────
    st.markdown("**§3 BEI + §5 Mitigation — Building-Level Analysis**")
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(cp.chart_eui_distribution(df_dem), width="stretch")
        st.markdown(_tip(
            "<b>Chart 5 — EUI Distribution &amp; Outlier Flagging</b> &nbsp;<i>SECAP §3 BEI + §5 Mitigation</i><br><br>"
            "<b>EUI</b> = Energy Use Intensity = total annual energy ÷ GFA, expressed in kWh/m²/yr — the standard metric for comparing buildings regardless of size<br>"
            "<b>GFA</b> = Gross Floor Area (m²)<br>"
            "<b>NZEB</b> = Near-Zero Energy Building — EU standard; typically ≤200 kWh/m²/yr<br><br>"
            "Histogram of EUI across all buildings. Buildings above the NZEB threshold are automatically labeled as priority retrofit targets."
        ), unsafe_allow_html=True)
        with st.expander("ℹ️ About this chart"):
            st.markdown(
                "**Chart 5 — EUI Distribution & Outlier Flagging** · *SECAP §3 BEI + §5 Mitigation*\n\n"
                "Histogram of Energy Use Intensity (kWh/m²/yr) across all buildings. "
                "A vertical threshold line marks the near-zero energy building (NZEB) target (200 kWh/m²/yr). "
                "Buildings above the threshold are labeled as priority retrofit candidates. "
                "This chart identifies the worst performers without requiring individual building inspection."
            )
    with r3c2:
        st.plotly_chart(cp.chart_solar_vs_demand(df_dem), width="stretch")
        st.markdown(_tip(
            "<b>Chart 6 — Solar PV Potential vs Total Demand</b> &nbsp;<i>SECAP §5 Mitigation</i><br><br>"
            "<b>PV</b> = Photovoltaic — panels that convert sunlight directly into electricity<br>"
            "<b>ST</b> = Solar Thermal — panels that use sunlight to heat water<br>"
            "<b>Energy gap</b> = demand minus on-site PV generation; what still needs to come from the grid or other sources<br><br>"
            "Per-building comparison of total demand vs rooftop PV potential, sorted by unfulfilled gap. Shows which buildings can approach self-sufficiency."
        ), unsafe_allow_html=True)
        with st.expander("ℹ️ About this chart"):
            st.markdown(
                "**Chart 6 — Solar PV Potential vs Total Demand** · *SECAP §5 Mitigation — Solar PV/ST*\n\n"
                "Per-building comparison of total annual energy demand against rooftop PV generation potential. "
                "Buildings are sorted by their energy gap (demand minus PV). The coverage % shows which buildings "
                "can approach self-sufficiency via solar and which require district-scale infrastructure regardless."
            )

    st.divider()

    # ── Seasonal + RVA: Charts 3, 4 ─────────────────────────────────────────
    if dfh_dem is not None:
        st.markdown("**§4 RVA — Seasonal & Heat Stress**")
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            st.plotly_chart(cp.chart_monthly_demand(dfh_dem), width="stretch")
            st.markdown(_tip(
                "<b>Chart 3 — Monthly Demand Profile</b> &nbsp;<i>SECAP §4 RVA + §5 Mitigation</i><br><br>"
                "<b>RVA</b> = Risk &amp; Vulnerability Assessment — SECAP section evaluating climate-related risks (heatwaves, flooding, etc.)<br>"
                "<b>Seasonal storage</b> = storing summer solar heat underground for winter use (e.g. aquifer thermal energy storage — ATES)<br><br>"
                "Monthly heating, cooling, and electricity totals across the year. Reveals seasonal imbalance and informs infrastructure timing strategies."
            ), unsafe_allow_html=True)
            with st.expander("ℹ️ About this chart"):
                st.markdown(
                    "**Chart 3 — Monthly Demand Profile** · *SECAP §4 RVA + §5 Mitigation*\n\n"
                    "Monthly aggregated demand for heating, cooling, and electricity across the full year. "
                    "Reveals seasonal imbalance and whether cross-seasonal solutions (e.g. aquifer thermal storage, "
                    "solar seasonal storage) are feasible. SECAP §4 requires characterisation of seasonal "
                    "vulnerability; §5 uses this to time mitigation actions."
                )
        with r4c2:
            fig_heat = cp.chart_heat_stress(dfh_dem)
            if fig_heat:
                st.plotly_chart(fig_heat, width="stretch")
                st.markdown(_tip(
                    "<b>Chart 4 — Heat Stress Risk Heatmap</b> &nbsp;<i>SECAP §4 RVA + §6 Adaptation</i><br><br>"
                    "<b>RVA</b> = Risk &amp; Vulnerability Assessment<br>"
                    "<b>Operative temperature</b> = the combined thermal sensation felt by occupants (air temp + radiant heat from surfaces)<br>"
                    "<b>§6 Adaptation</b> = SECAP section on adapting the built environment to future climate events: heatwaves, flooding, drought<br><br>"
                    "Hour × month heatmap of mean indoor operative temperature. Highlights when and how severely buildings overheat."
                ), unsafe_allow_html=True)
                with st.expander("ℹ️ About this chart"):
                    st.markdown(
                        "**Chart 4 — Heat Stress Risk Heatmap** · *SECAP §4 RVA + §6 Adaptation*\n\n"
                        "Pivot of mean operative temperature by hour-of-day × calendar month. "
                        "Reveals which months and daily hours place the heaviest passive cooling burden on the stock. "
                        "Directly populates SECAP §4 'Expected Climate Events: heatwaves' and informs §6 Adaptation "
                        "measures such as shading retrofits, ventilative cooling strategies, and green-roof interventions."
                    )
            else:
                st.info("Heat stress heatmap requires cooling demand data.")

        st.divider()

        # ── Infrastructure: Charts 2, 7 ─────────────────────────────────────
        st.markdown("**§5 Mitigation — Infrastructure Sizing & Heat Loop**")
        r5c1, r5c2 = st.columns(2)
        with r5c1:
            st.plotly_chart(cp.chart_load_duration(dfh_dem), width="stretch")
            st.markdown(_tip(
                "<b>Chart 2 — Load Duration Curves</b> &nbsp;<i>SECAP §5 Mitigation</i><br><br>"
                "<b>Load duration curve</b> = all 8,760 hours of the year sorted from highest to lowest demand — shows how often peak loads occur<br>"
                "<b>95th percentile</b> = the demand level exceeded only 5 % of the time; the correct design point to avoid costly over-sizing for rare peaks<br>"
                "<b>kWh/h</b> = kilowatt-hours per hour = kilowatts (kW) of instantaneous power demand<br><br>"
                "Used to correctly size district heating plants, cooling networks, and grid substations."
            ), unsafe_allow_html=True)
            with st.expander("ℹ️ About this chart"):
                st.markdown(
                    "**Chart 2 — Load Duration Curves** · *SECAP §5 Mitigation*\n\n"
                    "Each district load (space heating, cooling, grid electricity) is sorted from peak to minimum "
                    "across all 8,760 hours of the year. The 95th-percentile mark shows the capacity threshold "
                    "that covers 95% of all demand hours. This is the primary tool for sizing new energy "
                    "infrastructure (district heating plant, cooling network, grid upgrades) without over-building "
                    "for rare peak events — directly supporting the 'Estimated Energy Savings' metric in §5."
                )
        with r5c2:
            fig_loop = cp.chart_district_loop(dfh_dem)
            if fig_loop:
                st.plotly_chart(fig_loop, width="stretch")
                st.markdown(_tip(
                    "<b>Chart 7 — District Energy Loop Opportunity</b> &nbsp;<i>SECAP §5 Mitigation</i><br><br>"
                    "<b>District energy loop</b> = a shared pipe network that exchanges waste heat between buildings that are simultaneously heating and cooling<br>"
                    "<b>Heat pump cascade</b> = routing heat rejected by one building’s chiller as the input to another’s heat pump, boosting overall system efficiency<br>"
                    "<b>COP</b> = Coefficient of Performance — how many kWh of heat a heat pump delivers per kWh of electricity consumed (typical: 3–4)<br><br>"
                    "Quantifies co-occurring heating &amp; cooling hours and the recoverable MWh/yr — the evidence base for a district heat exchange loop action."
                ), unsafe_allow_html=True)
                with st.expander("ℹ️ About this chart"):
                    st.markdown(
                        "**Chart 7 — District Energy Loop Opportunity** · *SECAP §5 Mitigation*\n\n"
                        "Scatter of hourly heating vs cooling load, highlighting hours where both are non-zero. "
                        "Quantifies the waste-heat recovery potential in MWh/yr — these are hours where heat "
                        "rejected by chillers could supply space heating elsewhere, justifying a district energy "
                        "loop or heat pump cascade. Provides the evidence base and energy savings estimate for "
                        "a district heat exchange loop mitigation action in §5."
                    )

    st.divider()

    # ── CO₂ Pathway: Charts 8, D ────────────────────────────────────────────
    st.markdown("**§5 Mitigation — CO₂ Reduction Pathway**")
    st.plotly_chart(
        cp.chart_co2_waterfall(df_dem, kpis["total_co2"], _DEFAULT_EF),
        width="stretch",
    )
    st.markdown(_tip(
        "<b>Chart 8 — CO₂ Reduction Pathway (Waterfall)</b> &nbsp;<i>SECAP §5 Mitigation</i><br><br>"
        "<b>HP</b> = Heat Pump — electric device that moves heat instead of generating it; <b>COP</b> ≈3 means 3 kWh of heat per 1 kWh of electricity<br>"
        "<b>Grid decarbonization</b> = the German national grid emission factor is projected to fall from 0.366 to 0.200 tCO₂/MWh by 2030<br>"
        "<b>tCO₂e</b> = tonnes of CO₂-equivalent — the standard unit for measuring greenhouse gas emissions<br><br>"
        "Waterfall showing cumulative CO₂ savings from three interventions: fossil→HP electrification, grid clean-up, and rooftop solar PV."
    ), unsafe_allow_html=True)
    with st.expander("ℹ️ About this chart"):
        st.markdown(
            "**Chart 8 — CO₂ Reduction Pathway (Waterfall)** · *SECAP §5 Mitigation*\n\n"
            "Starting from the baseline CO₂, shows three intervention scenarios as incremental reductions: "
            "① **Fossil → Heat Pump Electrification** (replace all NG/OIL/COAL with electric HPs at COP=3), "
            "② **Grid Decarbonization** (German grid factor drops from 0.366 → 0.200 tCO₂/MWh by 2030), "
            "③ **Rooftop Solar PV** (full deployment of modelled PV potential offsets grid CO₂). "
            "The 2030 target line (−40%) is overlaid. This chart directly populates the SECAP §5 mitigation "
            "action table with estimated GHG reductions per intervention."
        )
    fig_timeline = cp.chart_fossil_timeline(
        kpis["fossil_mwh"], kpis["co2_fossil"], _DEFAULT_EF
    )
    if fig_timeline:
        st.plotly_chart(fig_timeline, width="stretch")
        st.markdown(_tip(
            "<b>Chart D — Fossil Fuel Phase-Out Timeline</b> &nbsp;<i>SECAP §5 Mitigation + §2.7 Monitoring</i><br><br>"
            "<b>Retrofit rate</b> = percentage of fossil heating systems (gas boilers, oil burners) replaced with heat pumps or district heating each year<br>"
            "<b>§2.7 Monitoring</b> = SECAP section requiring measurable annual milestones and progress reporting to the Covenant of Mayors<br>"
            "<b>−40 % by 2030</b> = legally binding emission reduction commitment<br><br>"
            "Shows which retrofit pace (5–20 %/yr) is needed to reach the 2030 target — the concrete ‘pace-of-change’ metric for monitoring."
        ), unsafe_allow_html=True)
        with st.expander("ℹ️ About this chart"):
            st.markdown(
                "**Chart D — Fossil Fuel Phase-Out Timeline** · *SECAP §5 Mitigation + §2.7 Monitoring*\n\n"
                "Projects the decline of fossil fuel consumption and CO₂ emissions from 2025 to 2035 under "
                "four retrofit rates (5%, 10%, 15%, 20% of heating systems replaced per year). Each line is "
                "a different ambition scenario. The 2030 target line shows which retrofit rate is needed to meet "
                "the 40% commitment, giving the plan a concrete *pace-of-change* metric for §2.7 monitoring."
            )

    

