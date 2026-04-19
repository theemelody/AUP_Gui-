from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import csv
import zipfile
import tempfile
import os
from io import BytesIO
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# CORS (for requests coming from the React dev server)
origins = [
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
    "http://1.0.0.127:5173",
    "http://localhost:3000",  # create-react-app default
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data loading (equivalent to Streamlit's load_data) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "OneNeighborhood.shp")
MAPBOX_TO_CEA_USE_TYPE_MAPPING_PATH = os.path.join(
    BASE_DIR, "mappings", "MAPBOX_TO_CEA_USE_TYPE1.csv"
)
CONSTRUCTION_TYPE_MAPPING_PATH = os.path.join(
    BASE_DIR, "mappings", "DE_CONSTRUCTION_TYPE_MAPPING.csv"
)

try:
    gdf = gpd.read_file(DATA_PATH)
    # Ensure we always serve WGS84 GeoJSON to the frontend map.
    if gdf.crs is None and not gdf.empty:
        minx, miny, maxx, maxy = gdf.total_bounds
        if minx > 180 or maxx > 180 or miny > 90 or maxy > 90:
            gdf = gdf.set_crs(epsg=32636)  # Turkey UTM 36N (meters)
        else:
            gdf = gdf.set_crs(epsg=4326)   # Already in degrees (WGS84)
    gdf = gdf.to_crs(epsg=4326)
except Exception as e:
    raise RuntimeError(f"Data could not be loaded: {e}")

class ChatRequest(BaseModel):
    # User chat prompt passed through to Ollama.
    message: str

class SelectRequest(BaseModel):
    # Geometry drawn on map (Feature, FeatureCollection, or geometry object).
    geometry: dict  # GeoJSON geometry


def load_mapbox_to_cea_use_type_mapping():
    """Load mapbox_type -> cea_use_type1 mapping from CSV used by the frontend."""
    if not os.path.exists(MAPBOX_TO_CEA_USE_TYPE_MAPPING_PATH):
        raise FileNotFoundError(
            f"Mapping file not found: {MAPBOX_TO_CEA_USE_TYPE_MAPPING_PATH}"
        )

    mapping = {}
    with open(MAPBOX_TO_CEA_USE_TYPE_MAPPING_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapbox_type = (row.get("mapbox_type") or "").strip().lower()
            cea_type = (row.get("cea_use_type1") or "").strip().upper()
            if mapbox_type and cea_type:
                mapping[mapbox_type] = cea_type

    return mapping


def load_construction_type_mapping():
    """Load decomposed construction types from CSV used by construction phase UI."""
    if not os.path.exists(CONSTRUCTION_TYPE_MAPPING_PATH):
        raise FileNotFoundError(
            f"Construction mapping file not found: {CONSTRUCTION_TYPE_MAPPING_PATH}"
        )

    def normalize_key(value: str) -> str:
        return (
            str(value or "")
            .replace("\ufeff", "")
            .strip()
            .lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )

    def read_row_value(row_dict: dict, aliases: list[str]) -> str:
        normalized_aliases = {normalize_key(alias) for alias in aliases}
        for key, value in row_dict.items():
            if normalize_key(key) in normalized_aliases:
                return str(value or "").strip()
        return ""

    rows = []
    with open(CONSTRUCTION_TYPE_MAPPING_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            const_type = read_row_value(row, ["const_type", "constType"])
            refurb = read_row_value(
                row,
                ["refurbishment_type", "refurbishmentType", "refurbishment"],
            )
            detail = read_row_value(row, ["detail", "detail_type", "detailType"])
            cea_use_type1 = read_row_value(
                row, ["cea_use_type1", "ceaUseType1", "use_type", "useType"]
            ).upper()
            if not (const_type and refurb and detail and cea_use_type1):
                continue
            try:
                year_start = int(read_row_value(row, ["year_start", "yearStart"]))
                year_end = int(read_row_value(row, ["year_end", "yearEnd"]))
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "const_type": const_type,
                    "year_start": year_start,
                    "year_end": year_end,
                    "refurbishment_type": refurb,
                    "detail": detail,
                    "cea_use_type1": cea_use_type1,
                }
            )

    return rows

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Proxy chat prompts to Ollama, supporting both /api/chat and /api/generate."""
    try:
        tags_resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        tags_resp.raise_for_status()
        payload_chat = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are an urban planning assistant."},
                {"role": "user", "content": req.message},
            ],
            "stream": False,
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload_chat, timeout=60)

        # Some Ollama setups expose /api/generate but not /api/chat.
        if resp.status_code == 404:
            payload_generate = {
                "model": OLLAMA_MODEL,
                "prompt": req.message,
                "stream": False,
            }
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload_generate,
                timeout=60,
            )

        resp.raise_for_status()
        data = resp.json()
        reply = data.get("message", {}).get("content") or data.get("response", "")
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama chat failed. Ensure `ollama serve` is running and model `{OLLAMA_MODEL}` exists. Raw error: {e}",
        )

@app.get("/api/buildings")
def get_buildings():
    """
    Returns all buildings as GeoJSON (for the map on the frontend).
    """
    return gdf.to_json()


@app.get("/api/mapbox-cea-use-type-mapping")
def get_mapbox_cea_use_type_mapping():
    """Returns CSV-based type mapping for frontend-side Mapbox selection."""
    try:
        return load_mapbox_to_cea_use_type_mapping()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load mapping CSV: {e}")


@app.get("/api/construction-type-mapping")
def get_construction_type_mapping():
    """Returns decomposed construction type mapping rows for frontend construction phase."""
    try:
        return load_construction_type_mapping()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load construction mapping CSV: {e}")

@app.post("/api/select")
def select_buildings(req: SelectRequest):
    """
    Finds buildings intersecting with the drawn geometry and returns:
    - selected building attributes
    - shapefile ZIP encoded as base64.
    """
    try:
        geo = req.geometry
        geo_type = geo.get("type") if isinstance(geo, dict) else None

        if geo_type == "FeatureCollection":
            feature_geoms = [
                shape(feature.get("geometry"))
                for feature in geo.get("features", [])
                if isinstance(feature, dict) and feature.get("geometry")
            ]
            if not feature_geoms:
                return {
                    "count": 0,
                    "buildings": [],
                    "selected_geojson": None,
                    "zip_base64": None,
                }
            drawn_geom = unary_union(feature_geoms)
        elif geo_type == "Feature":
            drawn_geom = shape(geo.get("geometry"))
        else:
            drawn_geom = shape(geo)

        selected = gdf[gdf.intersects(drawn_geom)]

        # Empty selection
        if selected.empty:
            return {
                "count": 0,
                "buildings": [],
                "selected_geojson": None,
                "zip_base64": None,
            }

        # Non-geometry columns from the GeoDataFrame
        df_props = selected.drop(columns="geometry")
        buildings = df_props.to_dict(orient="records")

        # Create shapefile ZIP
        tmpdir = tempfile.mkdtemp()
        shp_path = os.path.join(tmpdir, "selected_buildings.shp")
        selected.to_file(shp_path)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            for f in os.listdir(tmpdir):
                z.write(os.path.join(tmpdir, f), arcname=f)
        zip_buffer.seek(0)

        zip_b64 = base64.b64encode(zip_buffer.read()).decode("utf-8")

        return {
            "count": len(selected),
            "buildings": buildings,
            "selected_geojson": selected.to_json(),
            "zip_base64": zip_b64,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))