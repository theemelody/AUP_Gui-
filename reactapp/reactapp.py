import asyncio
import json
import csv
import datetime
import zipfile
import tempfile
import os
import base64
import requests
from io import BytesIO

import geopandas as gpd
from shapely.geometry import shape
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

# CORS (for requests coming from the React dev server)
_cors_env = os.getenv("CORS_ORIGINS", "")
origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
        "http://localhost:3000",
    ]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data loading (equivalent to Streamlit's load_data) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "frontend", ".env"), override=False)
MAPBOX_TO_CEA_USE_TYPE_MAPPING_PATH = os.path.join(
    BASE_DIR, "mappings", "MAPBOX_TO_CEA_USE_TYPE1.csv"
)
CONSTRUCTION_TYPE_MAPPING_PATH = os.path.join(
    BASE_DIR, "mappings", "DE_CONSTRUCTION_TYPE_MAPPING.csv"
)
CONSTRUCTION_TYPE_MAPPING_JSON_PATH = os.path.join(
    BASE_DIR, "mappings", "DE_CONSTRUCTION_TYPE_MAPPING.json"
)
CEA_DE_CONSTRUCTION_TYPES_PATH = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "CityEnergyAnalyst",
    "cea",
    "databases",
    "DE",
    "ARCHETYPES",
    "CONSTRUCTION",
    "CONSTRUCTION_TYPES.csv",
)
MAPBOX_REVERSE_GEOCODE_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"

CEA_DE_ASSEMBLIES_PATH = os.path.join(
    BASE_DIR, "..", "..", "CityEnergyAnalyst", "cea", "databases", "DE", "ASSEMBLIES"
)
CEA_DE_FEEDSTOCKS_PATH = os.path.join(
    BASE_DIR, "..", "..", "CityEnergyAnalyst", "cea", "databases", "DE",
    "COMPONENTS", "FEEDSTOCKS", "FEEDSTOCKS_LIBRARY"
)
CEA_DE_CONVERSION_PATH = os.path.join(
    BASE_DIR, "..", "..", "CityEnergyAnalyst", "cea", "databases", "DE",
    "COMPONENTS", "CONVERSION"
)

ARCHETYPE_FIELD_TO_ASSEMBLY = {
    "type_wall":       ("l2.1", "ENVELOPE_WALL"),
    "type_roof":       ("l2.1", "ENVELOPE_ROOF"),
    "type_win":        ("l2.1", "ENVELOPE_WINDOW"),
    "type_floor":      ("l2.1", "ENVELOPE_FLOOR"),
    "type_base":       ("l2.1", "ENVELOPE_FLOOR"),
    "type_shade":      ("l2.1", "ENVELOPE_SHADING"),
    "type_mass":       ("l2.1", "ENVELOPE_MASS"),
    "type_leak":       ("l2.1", "ENVELOPE_TIGHTNESS"),
    "hvac_type_hs":    ("l2.2", "HVAC_HEATING"),
    "hvac_type_cs":    ("l2.2", "HVAC_COOLING"),
    "hvac_type_dhw":   ("l2.2", "HVAC_HOTWATER"),
    "hvac_type_ctrl":  ("l2.2", "HVAC_CONTROLLER"),
    "hvac_type_vent":  ("l2.2", "HVAC_VENTILATION"),
    "supply_type_hs":  ("l2.3", "SUPPLY_HEATING"),
    "supply_type_dhw": ("l2.3", "SUPPLY_HOTWATER"),
    "supply_type_cs":  ("l2.3", "SUPPLY_COOLING"),
    "supply_type_el":  ("l2.3", "SUPPLY_ELECTRICITY"),
}

ASSEMBLY_FAMILY_TO_FILE = {
    "ENVELOPE_WALL":      ("ENVELOPE", "ENVELOPE_WALL.csv"),
    "ENVELOPE_ROOF":      ("ENVELOPE", "ENVELOPE_ROOF.csv"),
    "ENVELOPE_WINDOW":    ("ENVELOPE", "ENVELOPE_WINDOW.csv"),
    "ENVELOPE_FLOOR":     ("ENVELOPE", "ENVELOPE_FLOOR.csv"),
    "ENVELOPE_SHADING":   ("ENVELOPE", "ENVELOPE_SHADING.csv"),
    "ENVELOPE_MASS":      ("ENVELOPE", "ENVELOPE_MASS.csv"),
    "ENVELOPE_TIGHTNESS": ("ENVELOPE", "ENVELOPE_TIGHTNESS.csv"),
    "HVAC_HEATING":       ("HVAC",     "HVAC_HEATING.csv"),
    "HVAC_COOLING":       ("HVAC",     "HVAC_COOLING.csv"),
    "HVAC_HOTWATER":      ("HVAC",     "HVAC_HOTWATER.csv"),
    "HVAC_CONTROLLER":    ("HVAC",     "HVAC_CONTROLLER.csv"),
    "HVAC_VENTILATION":   ("HVAC",     "HVAC_VENTILATION.csv"),
    "SUPPLY_HEATING":     ("SUPPLY",   "SUPPLY_HEATING.csv"),
    "SUPPLY_HOTWATER":    ("SUPPLY",   "SUPPLY_HOTWATER.csv"),
    "SUPPLY_COOLING":     ("SUPPLY",   "SUPPLY_COOLING.csv"),
    "SUPPLY_ELECTRICITY": ("SUPPLY",   "SUPPLY_ELECTRICITY.csv"),
}

FUEL_CODE_TO_FEEDSTOCK = {
    "Cgas":   "NATURALGAS",
    "Coil":   "OIL",
    "Cdbm":   "DRYBIOMASS",
    "Cwod":   "WOOD",
    "E230AC": "GRID",
}

CONVERSION_FILE_TO_FEEDSTOCK = {
    "HEAT_PUMPS.csv":                  "GRID",
    "PHOTOVOLTAIC_PANELS.csv":         "SOLAR",
    "PHOTOVOLTAIC_THERMAL_PANELS.csv": "SOLAR",
    "SOLAR_COLLECTORS.csv":            "SOLAR",
    "FUEL_CELLS.csv":                  "GRID",
}

_techtree_logger = logging.getLogger("techtree")
NOMINATIM_REVERSE_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse"
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SCENARIOS_DIR = os.path.join(PROJECT_ROOT, "scenarios")
DEFAULT_SCENARIO_PATH = os.path.join(SCENARIOS_DIR, "output-scenario")

class ChatRequest(BaseModel):
    message: str
    model: str | None = None

class ExportCeaShpRequest(BaseModel):
    # Selected GeoJSON FeatureCollection created by the Mapbox selection pipeline.
    selected_geojson: dict
    scenario_name: str | None = None
    site_polygon: dict | None = None  # GeoJSON geometry object for site.shp (the area selection boundary)


def _normalize_key(value: str) -> str:
    return (
        str(value or "")
        .replace("\ufeff", "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def _read_row_value(row_dict: dict, aliases: list[str]) -> str:
    normalized_aliases = {_normalize_key(alias) for alias in aliases}
    for key, value in row_dict.items():
        if _normalize_key(key) in normalized_aliases:
            return str(value or "").strip()
    return ""


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
            mapbox_type = _read_row_value(row, ["mapbox_type", "mapboxType"]).lower()
            cea_type = _read_row_value(
                row,
                ["cea_use_type1", "ceaUseType1", "use_type", "useType"],
            ).upper()
            if mapbox_type and cea_type:
                mapping[mapbox_type] = cea_type

    return mapping


def load_construction_type_mapping():
    """Load decomposed construction types from JSON or CSV used by construction phase UI."""

    def normalize_mapbox_types(value):
        if value is None:
            return []
        if isinstance(value, list):
            raw_values = value
        else:
            raw_values = [item.strip() for item in str(value).split(",")]

        seen = set()
        normalized = []
        for item in raw_values:
            normalized_item = str(item or "").strip().lower()
            if not normalized_item or normalized_item in seen:
                continue
            seen.add(normalized_item)
            normalized.append(normalized_item)
        return normalized

    if os.path.exists(CONSTRUCTION_TYPE_MAPPING_JSON_PATH):
        with open(CONSTRUCTION_TYPE_MAPPING_JSON_PATH, encoding="utf-8") as f:
            payload = json.load(f)

        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Construction mapping JSON must contain a list of rows")

        normalized_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                year_start = int(row.get("year_start"))
                year_end = int(row.get("year_end"))
            except (TypeError, ValueError):
                continue

            const_type = str(row.get("const_type") or row.get("constType") or "").strip()
            refurb = str(
                row.get("refurbishment_type")
                or row.get("refurbishmentType")
                or row.get("refurbishment")
                or ""
            ).strip()
            detail = str(row.get("detail") or row.get("detail_type") or row.get("detailType") or "").strip()
            cea_use_type1 = str(
                row.get("cea_use_type1")
                or row.get("ceaUseType1")
                or row.get("use_type")
                or row.get("useType")
                or ""
            ).strip().upper()
            mapbox_types = normalize_mapbox_types(
                row.get("mapbox_type")
                or row.get("mapboxType")
                or row.get("mapbox_types")
                or row.get("mapboxTypes")
            )

            if not (const_type and refurb and detail and cea_use_type1):
                continue

            normalized_rows.append(
                {
                    "const_type": const_type,
                    "year_start": year_start,
                    "year_end": year_end,
                    "refurbishment_type": refurb,
                    "detail": detail,
                    "cea_use_type1": cea_use_type1,
                    "mapbox_type": mapbox_types,
                }
            )

        return normalized_rows

    if not os.path.exists(CONSTRUCTION_TYPE_MAPPING_PATH):
        raise FileNotFoundError(
            f"Construction mapping file not found: {CONSTRUCTION_TYPE_MAPPING_PATH}"
        )

    rows = []
    with open(CONSTRUCTION_TYPE_MAPPING_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            const_type = _read_row_value(row, ["const_type", "constType"])
            refurb = _read_row_value(
                row,
                ["refurbishment_type", "refurbishmentType", "refurbishment"],
            )
            detail = _read_row_value(row, ["detail", "detail_type", "detailType"])
            cea_use_type1 = _read_row_value(
                row, ["cea_use_type1", "ceaUseType1", "use_type", "useType"]
            ).upper()
            if not (const_type and refurb and detail and cea_use_type1):
                continue
            try:
                year_start = int(_read_row_value(row, ["year_start", "yearStart"]))
                year_end = int(_read_row_value(row, ["year_end", "yearEnd"]))
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
                    "mapbox_type": [],
                }
            )

    return rows


def load_cea_construction_rows():
    if not os.path.exists(CEA_DE_CONSTRUCTION_TYPES_PATH):
        return []

    rows = []
    with open(CEA_DE_CONSTRUCTION_TYPES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            const_type = str(row.get("const_type") or "").strip()
            if not const_type:
                continue
            try:
                year_start = int(float(row.get("year_start") or 0))
                year_end = int(float(row.get("year_end") or 0))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "const_type": const_type,
                    "year_start": year_start,
                    "year_end": year_end,
                    "description": str(row.get("description") or "").strip(),
                }
            )
    return rows


def normalize_scenario_name(value):
    raw_name = normalize_text(value)
    if not raw_name:
        return None, DEFAULT_SCENARIO_PATH

    folder_name = raw_name if raw_name.endswith("-scenario") else f"{raw_name}-scenario"
    return raw_name, os.path.join(SCENARIOS_DIR, folder_name)


def _build_zone_record(row) -> dict:
    """Build a CEA-4 compliant zone record dict from a GeoDataFrame row.

    Reference: cea/datamanagement/format_helper/cea4_verify.py COLUMNS_ZONE_4
    """
    return {
        "geometry":   row["geometry"],
        "name":       row["name"],
        "floors_bg":  int(row.get("floors_bg", 0)),
        "floors_ag":  int(row["floors_ag"]),
        "void_deck":  int(row.get("void_deck", 0)),
        "height_bg":  float(row.get("height_bg", 0.0)),
        "height_ag":  float(row["height_ag"]),
        "year":       int(row.get("year", 2000)),
        "const_type": str(row.get("const_type", "")),
        "use_type1":  str(row.get("use_type1", "UNKNOWN")),
        "use_type1r": float(row.get("use_type1r", 1.0)),
        "use_type2":  str(row.get("use_type2", "NONE")),
        "use_type2r": float(row.get("use_type2r", 0.0)),
        "use_type3":  str(row.get("use_type3", "NONE")),
        "use_type3r": float(row.get("use_type3r", 0.0)),
    }


def prepare_scenario_structure(scenario_path, zone_gdf, site_polygon=None):
    """Create scenario directory structure and write zone/site shapefiles.

    site_polygon: GeoJSON geometry object for the area selection boundary.
                 If None, uses union of buildings.
    """
    os.makedirs(os.path.join(scenario_path, "inputs", "building-geometry"), exist_ok=True)
    os.makedirs(os.path.join(scenario_path, "inputs", "building-properties"), exist_ok=True)
    os.makedirs(os.path.join(scenario_path, "inputs", "topography"), exist_ok=True)
    os.makedirs(os.path.join(scenario_path, "inputs", "weather"), exist_ok=True)
    os.makedirs(os.path.join(scenario_path, "outputs"), exist_ok=True)

    geom_dir = os.path.join(scenario_path, "inputs", "building-geometry")

    zone_data = [_build_zone_record(row) for _, row in zone_gdf.iterrows()]
    zone_gdf_cleaned = gpd.GeoDataFrame(zone_data, crs=zone_gdf.crs)

    # Write zone.shp
    zone_path = os.path.join(geom_dir, "zone.shp")
    zone_gdf_cleaned.to_file(zone_path, encoding="UTF-8")

    # Use provided site polygon or fall back to union of buildings
    if site_polygon is not None:
        try:
            site_geom = shape(site_polygon)
        except Exception:
            site_geom = zone_gdf_cleaned.geometry.union_all()
    else:
        site_geom = zone_gdf_cleaned.geometry.union_all()

    site_gdf = gpd.GeoDataFrame(geometry=[site_geom], crs=zone_gdf.crs)
    site_gdf.to_file(os.path.join(geom_dir, "site.shp"), encoding="UTF-8")


def normalize_text(value):
    return str(value or "").strip()


def normalize_lower(value):
    return normalize_text(value).lower()


def normalize_upper(value):
    return normalize_text(value).upper()


def pick_first_value(payload, keys, default=None):
    for key in keys:
        value = payload.get(key)
        if value is not None and normalize_text(value):
            return value
    return default


def get_mapbox_building_type(properties):
    candidates = [
        properties.get("class"),
        properties.get("building"),
        properties.get("subclass"),
        properties.get("type"),
    ]

    for candidate in candidates:
        value = normalize_lower(candidate)
        if not value:
            continue
        if value == "building":
            continue
        return value

    fallback = normalize_lower(
        properties.get("type") or properties.get("class") or properties.get("building") or ""
    )
    return fallback or None


def get_building_feature_key(feature):
    properties = feature.get("properties") or {}
    building_id = properties.get("building_id") or properties.get("buildingId")
    if building_id not in (None, ""):
        return str(building_id)

    feature_id = feature.get("id")
    if feature_id not in (None, ""):
        return str(feature_id)

    return None


def build_building_type_lookup(features):
    lookup = {}

    for feature in features:
        properties = feature.get("properties") or {}
        feature_key = get_building_feature_key(feature)
        if not feature_key:
            continue

        mapbox_type = get_mapbox_building_type(properties)
        if mapbox_type and mapbox_type not in {"building", "building:part"}:
            lookup[feature_key] = mapbox_type

    return lookup


def normalize_feature_collection(selected_geojson):
    if not isinstance(selected_geojson, dict):
        raise ValueError("selected_geojson must be a GeoJSON FeatureCollection or Feature")

    geo_type = selected_geojson.get("type")
    if geo_type == "FeatureCollection":
        features = [
            feature
            for feature in selected_geojson.get("features", [])
            if isinstance(feature, dict) and feature.get("geometry")
        ]
    elif geo_type == "Feature" and selected_geojson.get("geometry"):
        features = [selected_geojson]
    elif selected_geojson.get("geometry"):
        features = [{"type": "Feature", "properties": {}, "geometry": selected_geojson}]
    else:
        raise ValueError("selected_geojson does not contain any feature geometry")

    if not features:
        raise ValueError("selected_geojson does not contain any feature geometry")

    return features


def geometry_centroid_lon_lat(geometry_obj):
    geom = shape(geometry_obj)
    centroid = geom.centroid
    return float(centroid.x), float(centroid.y)


def reverse_geocode_mapbox(lon, lat):
    token = (
        os.getenv("MAPBOX_ACCESS_TOKEN")
        or os.getenv("VITE_MAPBOX_ACCESS_TOKEN")
        or os.getenv("MAPBOX_TOKEN")
    )
    if not token:
        return None

    url = MAPBOX_REVERSE_GEOCODE_URL.format(lon=lon, lat=lat)
    params = {
        "types": "address,postcode,place,poi",
        "limit": 1,
        "language": "en",
        "access_token": token,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features") or []
    if not features:
        return None
    feature = features[0] if isinstance(features[0], dict) else {}
    context = feature.get("context") or []
    address_number = normalize_text(feature.get("address"))
    street = normalize_text(feature.get("text"))
    postcode = ""
    city = ""
    country = ""
    house_name = normalize_text(feature.get("place_name"))

    if isinstance(context, list):
        for item in context:
            if not isinstance(item, dict):
                continue
            item_id = normalize_lower(item.get("id"))
            item_text = normalize_text(item.get("text"))
            if item_id.startswith("postcode") and not postcode:
                postcode = item_text
            elif item_id.startswith(("place", "locality", "district")) and not city:
                city = item_text
            elif item_id.startswith("country") and not country:
                country = item_text

    return {
        "house_name": house_name,
        "house_no": address_number,
        "street": street,
        "postcode": postcode,
        "city": city,
        "country": country,
        "reference": "Mapbox reverse geocoding",
    }


def reverse_geocode_osm(lon, lat):
    params = {
        "format": "jsonv2",
        "lon": lon,
        "lat": lat,
        "addressdetails": 1,
        "zoom": 18,
    }
    headers = {
        "User-Agent": "automatic-urban-planner-reactapp/1.0 (CEA export pipeline)",
        "Accept-Language": "en",
    }
    resp = requests.get(NOMINATIM_REVERSE_GEOCODE_URL, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    address = data.get("address") if isinstance(data, dict) else {}
    if not isinstance(address, dict):
        address = {}
    road = normalize_text(address.get("road") or address.get("pedestrian") or address.get("footway"))
    house_no = normalize_text(address.get("house_number"))
    postcode = normalize_text(address.get("postcode"))
    city = normalize_text(
        address.get("city") or address.get("town") or address.get("village") or address.get("municipality")
    )
    country = normalize_text(address.get("country"))
    house_name = normalize_text(data.get("name") or data.get("display_name"))

    return {
        "house_name": house_name,
        "house_no": house_no,
        "street": road,
        "postcode": postcode,
        "city": city,
        "country": country,
        "reference": "OSM reverse geocoding",
    }


def get_address_data_for_feature(feature):
    geometry_obj = feature.get("geometry") or {}
    lon, lat = geometry_centroid_lon_lat(geometry_obj)
    cache_key = (round(lon, 6), round(lat, 6))
    if not hasattr(get_address_data_for_feature, "_cache"):
        get_address_data_for_feature._cache = {}
    cache = get_address_data_for_feature._cache
    if cache_key in cache:
        return cache[cache_key]

    candidates = []
    try:
        candidates.append(reverse_geocode_mapbox(lon, lat))
    except Exception:
        candidates.append(None)

    try:
        candidates.append(reverse_geocode_osm(lon, lat))
    except Exception:
        candidates.append(None)

    result = next((item for item in candidates if item), {})
    cache[cache_key] = result
    return result


def build_projection_crs(gdf):
    if gdf.empty:
        return "EPSG:4326"

    wgs84 = gdf.to_crs(epsg=4326)
    centroid = wgs84.union_all().centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def choose_construction_family(use_type1, floors_ag):
    normalized_use_type = normalize_upper(use_type1)

    if not normalized_use_type:
        if floors_ag is not None and floors_ag > 2:
            return "NWG_G1"
        return "SFH"

    if "RES" in normalized_use_type:
        return "MFH" if floors_ag is not None and floors_ag > 2 else "SFH"
    if any(keyword in normalized_use_type for keyword in ["OFFICE", "ADMIN", "GOV", "WORK"]):
        return "NWG_1"
    if any(keyword in normalized_use_type for keyword in ["LAB", "RESEARCH", "UNIVERSITY"]):
        return "NWG_2"
    if any(keyword in normalized_use_type for keyword in ["HEALTH", "HOSP", "CARE"]):
        return "NWG_3"
    if any(keyword in normalized_use_type for keyword in ["SCHOOL", "EDUC", "DAYCARE"]):
        return "NWG_4"
    if any(keyword in normalized_use_type for keyword in ["CULTURE", "LEISURE", "MUSEUM", "CINEMA"]):
        return "NWG_5"
    if any(keyword in normalized_use_type for keyword in ["SPORT", "GYM", "STADIUM"]):
        return "NWG_6"
    if any(keyword in normalized_use_type for keyword in ["HOTEL", "RESTAUR", "CATER", "HOSPITALITY"]):
        return "NWG_7"
    if any(keyword in normalized_use_type for keyword in ["INDUSTR", "PROD", "WAREHOUSE", "WORKSHOP", "LOGISTIC"]):
        return "NWG_8"
    if any(keyword in normalized_use_type for keyword in ["TRADE", "RETAIL", "SHOP", "COMMER", "STORE"]):
        return "NWG_9"
    if any(keyword in normalized_use_type for keyword in ["UTILITY", "SERVICE", "TECH", "INFRA"]):
        return "NWG_10"
    if any(keyword in normalized_use_type for keyword in ["TRANSPORT", "STATION", "PARKING", "DEPOT"]):
        return "NWG_11"

    return "NWG_G1" if floors_ag is not None and floors_ag > 2 else "MFH"


def choose_const_type(use_type1, year, floors_ag):
    rows = load_cea_construction_rows()
    if not rows:
        return "MFH_I"

    family = choose_construction_family(use_type1, floors_ag)
    year = 2000 if year is None else int(year)

    family_rows = [row for row in rows if row["const_type"].startswith(family)]
    if not family_rows:
        family_rows = rows

    matching = [row for row in family_rows if row["year_start"] <= year <= row["year_end"]]
    if matching:
        matching.sort(key=lambda row: (row["year_start"], row["year_end"]))
        return matching[0]["const_type"]

    family_rows.sort(key=lambda row: (abs(row["year_start"] - year), abs(row["year_end"] - year)))
    return family_rows[0]["const_type"]


def coerce_number(value, default=None, minimum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        number = max(number, minimum)
    return number


def build_cea_export_records(features):
    mapping = load_mapbox_to_cea_use_type_mapping()
    type_lookup = build_building_type_lookup(features)
    records = []
    for index, feature in enumerate(features):
        properties = feature.get("properties") or {}
        feature_key = get_building_feature_key(feature)
        raw_mapbox_type = normalize_lower(
            pick_first_value(
                properties,
                ["mapbox_type_raw", "mapboxTypeRaw", "building_type_raw", "buildingTypeRaw"],
                "",
            )
        )
        if not raw_mapbox_type:
            raw_mapbox_type = get_mapbox_building_type(properties) or ""

        resolved_mapbox_type = type_lookup.get(feature_key) if feature_key else None
        if not resolved_mapbox_type and raw_mapbox_type not in {"building", "building:part", ""}:
            resolved_mapbox_type = raw_mapbox_type

        effective_mapbox_type = resolved_mapbox_type or raw_mapbox_type
        address_data = get_address_data_for_feature(feature)
        merged = {
            **address_data,
            "name": normalize_text(pick_first_value(properties, ["name", "Name", "building_name"], "")),
            "reference": normalize_text(
                pick_first_value(properties, ["reference", "REFERENCE"], address_data.get("reference", "Mapbox / OSM"))
            ) or address_data.get("reference", "Mapbox / OSM"),
            "cea_use_type1": normalize_upper(
                pick_first_value(
                    properties,
                    ["cea_use_type1", "ceaUseType1", "use_type1", "useType1"],
                    mapping.get(effective_mapbox_type, "UNKNOWN"),
                )
            ),
            "mapbox_type": normalize_lower(
                effective_mapbox_type
                or pick_first_value(properties, ["mapbox_type", "mapboxType", "class", "type", "building"], "")
            ),
            "mapbox_type_raw": raw_mapbox_type,
        }

        if not merged["name"] or not merged["name"][0].isalpha():
            merged["name"] = f"B{index + 1001}"

        floors_ag = coerce_number(
            pick_first_value(properties, ["floors_ag", "floorsAG", "estimated_floors", "levels", "num_floors"], None),
            default=None,
            minimum=1,
        )
        height_ag = coerce_number(
            pick_first_value(properties, ["height_ag", "height", "building_height"], None),
            default=None,
            minimum=None,
        )
        min_height = coerce_number(pick_first_value(properties, ["min_height"], 0), default=0, minimum=0)

        if floors_ag is None:
            if height_ag is not None and height_ag > min_height:
                floors_ag = max(1, round((height_ag - min_height) / 3.0))
            else:
                floors_ag = 3
        floors_ag = int(max(1, round(floors_ag)))

        if height_ag is None or height_ag <= min_height:
            height_ag = float(max(3.0, floors_ag * 3.0))

        year_value = pick_first_value(
            properties,
            ["year", "construction_year", "feature_year_start", "feature_year_end"],
            None,
        )
        if year_value is None:
            year = 2000
        else:
            try:
                if isinstance(year_value, str) and "-" in year_value:
                    start_year, end_year = [int(part) for part in year_value.split("-", 1)]
                    year = int((start_year + end_year) / 2)
                else:
                    year = int(float(year_value))
            except (TypeError, ValueError):
                year = 2000

        const_type = normalize_text(pick_first_value(properties, ["const_type", "constType"], ""))
        if not const_type:
            const_type = choose_const_type(merged["cea_use_type1"], year, floors_ag)

        merged.update(
            {
                "floors_bg": int(max(0, round(coerce_number(pick_first_value(properties, ["floors_bg"], 0), default=0, minimum=0)))),
                "floors_ag": floors_ag,
                "void_deck": int(max(0, round(coerce_number(pick_first_value(properties, ["void_deck"], 0), default=0, minimum=0)))),
                "height_bg": float(max(0.0, coerce_number(pick_first_value(properties, ["height_bg"], 0), default=0.0, minimum=0.0))),
                "height_ag": float(max(float(floors_ag), float(height_ag))),
                "year": year,
                "const_type": const_type,
                "use_type1": merged["cea_use_type1"],
                "use_type1r": float(coerce_number(pick_first_value(properties, ["use_type1r"], 1.0), default=1.0, minimum=0.0)),
                "use_type2": normalize_upper(pick_first_value(properties, ["use_type2"], "NONE")) or "NONE",
                "use_type2r": float(coerce_number(pick_first_value(properties, ["use_type2r"], 0.0), default=0.0, minimum=0.0)),
                "use_type3": normalize_upper(pick_first_value(properties, ["use_type3"], "NONE")) or "NONE",
                "use_type3r": float(coerce_number(pick_first_value(properties, ["use_type3r"], 0.0), default=0.0, minimum=0.0)),
                "house_no": normalize_text(pick_first_value(properties, ["house_no", "houseNumber", "number"], merged.get("house_no", ""))),
                "street": normalize_text(pick_first_value(properties, ["street", "road", "addr:street"], merged.get("street", ""))),
                "postcode": normalize_text(pick_first_value(properties, ["postcode", "postal_code", "zip"], merged.get("postcode", ""))),
                "house_name": normalize_text(pick_first_value(properties, ["house_name", "name", "display_name"], merged.get("house_name", ""))),
                "resi_type": normalize_text(pick_first_value(properties, ["resi_type"], "UNKNOWN")) or "UNKNOWN",
                "city": normalize_text(pick_first_value(properties, ["city", "place"], merged.get("city", ""))),
                "country": normalize_text(pick_first_value(properties, ["country"], merged.get("country", ""))),
            }
        )

        if not merged["street"] and merged["house_name"]:
            merged["street"] = merged["house_name"]

        records.append({"geometry": shape(feature.get("geometry")), **merged})

    return records


def write_cea_zip_from_geojson(features):
    export_records = build_cea_export_records(features)
    export_gdf = gpd.GeoDataFrame(export_records, crs="EPSG:4326")
    projected_crs = build_projection_crs(export_gdf)
    export_gdf = export_gdf.to_crs(projected_crs)

    tmpdir = tempfile.mkdtemp(prefix="cea-export-")

    zone_data = [_build_zone_record(row) for _, row in export_gdf.iterrows()]
    zone_gdf = gpd.GeoDataFrame(zone_data, crs=projected_crs)

    # Write zone.shp
    shp_path = os.path.join(tmpdir, "zone.shp")
    zone_gdf.to_file(shp_path, encoding="UTF-8")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in os.listdir(tmpdir):
            archive.write(os.path.join(tmpdir, filename), arcname=filename)
    zip_buffer.seek(0)
    zip_b64 = base64.b64encode(zip_buffer.read()).decode("utf-8")

    return export_gdf.to_crs(epsg=4326), zip_b64

def load_techtree_graph(region: str = "DE", scenario_name: str = "") -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_set: set[tuple] = set()

    def add_node(node_id, label, layer, sublayer, description=""):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "layer": layer, "sublayer": sublayer, "description": description}

    def add_edge(src, tgt, field=""):
        key = (src, tgt)
        if key not in edge_set:
            edge_set.add(key)
            edges.append({"id": f"e_{src}__{tgt}", "source": src, "target": tgt, **({"field": field} if field else {})})

    assemblies_base = os.path.join(BASE_DIR, "..", "..", "CityEnergyAnalyst", "cea", "databases", region, "ASSEMBLIES")
    conversion_base = os.path.join(BASE_DIR, "..", "..", "CityEnergyAnalyst", "cea", "databases", region, "COMPONENTS", "CONVERSION")
    feedstocks_base = os.path.join(BASE_DIR, "..", "..", "CityEnergyAnalyst", "cea", "databases", region, "COMPONENTS", "FEEDSTOCKS", "FEEDSTOCKS_LIBRARY")
    const_types_path = os.path.join(BASE_DIR, "..", "..", "CityEnergyAnalyst", "cea", "databases", region, "ARCHETYPES", "CONSTRUCTION", "CONSTRUCTION_TYPES.csv")

    extra_desc_path = os.path.join(BASE_DIR, "mappings", "TECHTREE_EXTRA_DESCRIPTIONS.json")
    extra_descriptions: dict = {}
    if os.path.exists(extra_desc_path):
        with open(extra_desc_path, encoding="utf-8") as _f:
            extra_descriptions = json.load(_f)

    # L0 + L1 nodes from CONSTRUCTION_TYPES.csv
    if os.path.exists(const_types_path):
        with open(const_types_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ct = str(row.get("const_type") or "").strip()
                if not ct:
                    continue
                desc = str(row.get("description") or "").strip()
                add_node(ct, ct, 0, "l0", desc)
                for field, (sublayer, family_id) in ARCHETYPE_FIELD_TO_ASSEMBLY.items():
                    code = str(row.get(field) or "").strip()
                    if not code:
                        continue
                    add_node(code, code, 1, "l1")
                    add_edge(ct, code, field)
    else:
        _techtree_logger.warning("TechTree: CONSTRUCTION_TYPES.csv not found at %s", const_types_path)

    # L2 nodes + L1→L2 edges, and L3 nodes from assembly files
    seen_families: set[str] = set()
    for family_id, (subcat, filename) in ASSEMBLY_FAMILY_TO_FILE.items():
        sublayer = "l2.1" if subcat == "ENVELOPE" else ("l2.2" if subcat == "HVAC" else "l2.3")
        layer2_rank = 2 if subcat == "ENVELOPE" else (3 if subcat == "HVAC" else 4)
        if family_id not in seen_families:
            seen_families.add(family_id)
            label = family_id.replace("_", " ").title()
            add_node(family_id, label, layer2_rank, sublayer)

        filepath = os.path.join(assemblies_base, subcat, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            has_primary = "primary_components" in headers
            has_feedstock = "feedstock" in headers
            for row in reader:
                code = str(row.get("code") or "").strip()
                if not code:
                    continue
                if code in nodes:
                    add_edge(code, family_id)
                    if not nodes[code]["description"]:
                        nodes[code]["description"] = str(row.get("description") or "").strip()
                if has_primary:
                    comp = str(row.get("primary_components") or "").strip()
                    if comp and comp != "-":
                        add_node(comp, comp, 5, "l3.1")
                        add_edge(family_id, comp)
                if has_feedstock:
                    fs = str(row.get("feedstock") or "").strip().upper()
                    if fs and fs != "NONE":
                        add_node(fs, fs.capitalize(), 6, "l3.2")
                        add_edge(family_id, fs)

    # L3.1→L3.2 edges from CONVERSION files
    if os.path.exists(conversion_base):
        for fname in os.listdir(conversion_base):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(conversion_base, fname)
            inferred_fs = CONVERSION_FILE_TO_FEEDSTOCK.get(fname)
            with open(fpath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                has_fuel = "fuel_code" in headers
                for row in reader:
                    code = str(row.get("code") or "").strip()
                    if not code or code not in nodes:
                        continue
                    if not nodes[code]["description"]:
                        nodes[code]["description"] = str(row.get("description") or "").strip()
                    if has_fuel:
                        fc = str(row.get("fuel_code") or "").strip()
                        fs = FUEL_CODE_TO_FEEDSTOCK.get(fc)
                    else:
                        fs = inferred_fs
                    if fs:
                        add_node(fs, fs.capitalize(), 6, "l3.2")
                        add_edge(code, fs)

    # Apply extra descriptions as fallback for nodes still missing a description
    for node_id, node in nodes.items():
        if not node["description"]:
            node["description"] = extra_descriptions.get(node_id, "")

    # Active const_types from scenario zone.shp
    active_const_types: list[str] = []
    if scenario_name:
        zone_path = os.path.join(SCENARIOS_DIR, f"{scenario_name}-scenario", "inputs", "building-geometry", "zone.shp")
        if os.path.exists(zone_path):
            try:
                zone_gdf = gpd.read_file(zone_path)
                active_const_types = list({str(v) for v in zone_gdf.get("const_type", []) if v and str(v).strip()})
            except Exception as exc:
                _techtree_logger.warning("TechTree: failed to read zone.shp for %s: %s", scenario_name, exc)

    node_list = list(nodes.values())
    _techtree_logger.info(
        "TechTree: %d L0, %d L1, %d L2, %d L3 nodes, %d edges",
        sum(1 for n in node_list if n["sublayer"] == "l0"),
        sum(1 for n in node_list if n["sublayer"] == "l1"),
        sum(1 for n in node_list if n["sublayer"].startswith("l2")),
        sum(1 for n in node_list if n["sublayer"].startswith("l3")),
        len(edges),
    )
    missing = [n["id"] for n in node_list if n["sublayer"] == "l1" and n["id"] not in {e["source"] for e in edges if e.get("field")}]
    if missing:
        _techtree_logger.warning("TechTree: %d L1 codes have no L1→L2 edge: %s", len(missing), missing[:10])

    return {"active_const_types": active_const_types, "nodes": node_list, "edges": edges}


@app.get("/api/techtree-graph")
def get_techtree_graph(region: str = "DE", scenario_name: str = ""):
    """Return full 4-layer construction hierarchy graph with active const_types from scenario."""
    try:
        return load_techtree_graph(region=region, scenario_name=scenario_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TechTree graph failed: {exc}")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Proxy chat prompts to Ollama, supporting both /api/chat and /api/generate."""
    try:
        tags_resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        tags_resp.raise_for_status()
        model = req.model or OLLAMA_MODEL
        payload_chat = {
            "model": model,
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
                "model": model,
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
            detail=f"Ollama chat failed. Ensure `ollama serve` is running and model `{model}` exists. Raw error: {e}",
        )


@app.get("/api/ollama-models")
def get_ollama_models():
    """Return list of locally available Ollama model names."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in (data.get("models") or []) if m.get("name")]
        return {"models": models or [OLLAMA_MODEL]}
    except Exception:
        return {"models": [OLLAMA_MODEL]}


@app.get("/api/scenario-status/{scenario_name}")
def get_scenario_status(scenario_name: str):
    """Check CEA readiness for a saved scenario."""
    folder = f"{scenario_name}-scenario"
    path = os.path.join(SCENARIOS_DIR, folder)
    if not os.path.isdir(path):
        return {"status": "missing"}
    zone_shp = os.path.join(path, "inputs", "building-geometry", "zone.shp")
    demand_csv = os.path.join(path, "outputs", "data", "demand", "Total_demand.csv")
    if os.path.exists(demand_csv):
        return {"status": "complete"}
    if os.path.exists(zone_shp):
        return {"status": "ready"}
    return {"status": "incomplete"}


CEA_CMD = os.environ.get("CEA_CMD", "/home/salva/micromamba/envs/cea/bin/cea")

SIMULATION_STEPS = [
    ("database-helper",     [CEA_CMD, "database-helper",     "--databases-path", "DE"]),
    ("archetypes-mapper",   [CEA_CMD, "archetypes-mapper"]),
    ("surroundings-helper", [CEA_CMD, "surroundings-helper"]),
    ("terrain-helper",      [CEA_CMD, "terrain-helper"]),
    ("weather-helper",      [CEA_CMD, "weather-helper"]),
    ("radiation",           [CEA_CMD, "radiation",           "--multiprocessing", "true"]),
    ("occupancy",           [CEA_CMD, "occupancy",           "--multiprocessing", "true"]),
    ("demand",              [CEA_CMD, "demand",              "--multiprocessing", "true"]),
]


_logger = logging.getLogger("cea_pipeline")


async def _cea_pipeline(scenario_path: str):
    base_args = ["--scenario", scenario_path]
    for step_name, cmd in SIMULATION_STEPS:
        full_cmd = cmd + base_args
        _logger.info("CEA step [%s]: %s", step_name, " ".join(full_cmd))
        yield f"data: {json.dumps({'step': step_name, 'status': 'running'})}\n\n"

        try:
            proc = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            msg = f"Executable not found: {full_cmd[0]} — {exc}"
            _logger.error("CEA step [%s] launch failed: %s", step_name, msg)
            yield f"data: {json.dumps({'step': step_name, 'status': 'error', 'message': msg})}\n\n"
            yield f"data: {json.dumps({'status': 'failed'})}\n\n"
            return

        tail: list[str] = []
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            _logger.debug("  [%s] %s", step_name, line)
            tail = (tail + [line])[-20:]  # keep last 20 lines for error context
            yield f"data: {json.dumps({'step': step_name, 'status': 'running', 'message': line})}\n\n"

        await proc.wait()

        if proc.returncode != 0:
            context = " | ".join(tail[-5:]) if tail else "(no output)"
            _logger.error(
                "CEA step [%s] exited %d. Last output: %s",
                step_name, proc.returncode, context,
            )
            yield f"data: {json.dumps({'step': step_name, 'status': 'error', 'message': f'Exit code {proc.returncode}: {context}'})}\n\n"
            yield f"data: {json.dumps({'status': 'failed'})}\n\n"
            return

        _logger.info("CEA step [%s] done.", step_name)
        yield f"data: {json.dumps({'step': step_name, 'status': 'done'})}\n\n"

    yield f"data: {json.dumps({'status': 'complete'})}\n\n"


@app.get("/api/run-simulation")
async def run_simulation(scenario_name: str):
    """Stream CEA simulation pipeline progress via SSE."""
    folder = f"{scenario_name}-scenario"
    scenario_path = os.path.join(SCENARIOS_DIR, folder)
    if not os.path.isdir(scenario_path):
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_path}")
    return StreamingResponse(
        _cea_pipeline(scenario_path),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

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


@app.post("/api/export-cea-shp")
def export_cea_shp(req: ExportCeaShpRequest):
    """Convert selected Mapbox GeoJSON into a CEA-compatible shapefile ZIP."""
    try:
        features = normalize_feature_collection(req.selected_geojson)
        export_gdf, zip_b64 = write_cea_zip_from_geojson(features)
        scenario_raw_name, scenario_path = normalize_scenario_name(req.scenario_name)
        if scenario_raw_name:
            prepare_scenario_structure(scenario_path, export_gdf, req.site_polygon)
        buildings = export_gdf.drop(columns="geometry").to_dict(orient="records")

        return {
            "count": len(export_gdf),
            "buildings": buildings,
            "selected_geojson": export_gdf.to_json(),
            "zip_base64": zip_b64,
            "filename": "cea_selected_buildings.zip",
            "scenario_name": scenario_raw_name,
            "scenario_path": scenario_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CEA export failed: {e}")

@app.get("/api/scenarios")
def list_scenarios():
    """List all scenario folders present in the scenarios directory."""
    try:
        if not os.path.isdir(SCENARIOS_DIR):
            return {"scenarios": []}
        names = []
        for entry in sorted(os.scandir(SCENARIOS_DIR), key=lambda e: e.name):
            if entry.is_dir():
                name = entry.name
                if name.endswith("-scenario"):
                    name = name[: -len("-scenario")]
                names.append(name)
        return {"scenarios": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list scenarios: {e}")


@app.post("/api/save-scenario")
def save_scenario(req: ExportCeaShpRequest):
    """Save building selection as uncompressed files to scenarios folder."""
    try:
        if not req.scenario_name or not req.scenario_name.strip():
            raise ValueError("Scenario name is required")
        
        features = normalize_feature_collection(req.selected_geojson)
        export_records = build_cea_export_records(features)
        export_gdf = gpd.GeoDataFrame(export_records, crs="EPSG:4326")
        projected_crs = build_projection_crs(export_gdf)
        export_gdf = export_gdf.to_crs(projected_crs)
        
        scenario_raw_name, scenario_path = normalize_scenario_name(req.scenario_name)
        prepare_scenario_structure(scenario_path, export_gdf, req.site_polygon)

        snapshot = {
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scenario_name": scenario_raw_name,
            "selected_geojson": req.selected_geojson,
            "drawn_polygon": req.site_polygon,
        }
        with open(os.path.join(scenario_path, "scenario.json"), "w", encoding="utf-8") as _f:
            json.dump(snapshot, _f)

        return {
            "success": True,
            "count": len(export_gdf),
            "scenario_name": scenario_raw_name,
            "scenario_path": scenario_path,
            "message": f"Scenario '{scenario_raw_name}' saved successfully with {len(export_gdf)} buildings"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save scenario failed: {e}")


@app.get("/api/scenario-data/{scenario_name}")
def get_scenario_data(scenario_name: str):
    """Return the scenario.json snapshot for map/TechTree restore."""
    _, scenario_path = normalize_scenario_name(scenario_name)
    snapshot_path = os.path.join(scenario_path, "scenario.json")
    if not os.path.exists(snapshot_path):
        raise HTTPException(status_code=404, detail="scenario.json not found")
    with open(snapshot_path, "r", encoding="utf-8") as _f:
        return json.load(_f)


