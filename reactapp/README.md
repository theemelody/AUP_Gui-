# ReactApp Workspace

This folder contains the React + FastAPI prototype used for map-based building selection and CEA-oriented classification.

## Folder Structure

- `frontend/`
  - React app (Vite) for map interaction, selection, and UI panels.
- `mappings/`
  - CSV mapping files used by the frontend/backend workflow.
  - `MAPBOX_TO_CEA_USE_TYPE1.csv`: maps Mapbox building categories to CEA `use_type1`.
  - `DE_CONSTRUCTION_TYPE_MAPPING.csv`: decomposed CEA construction types with `cea_use_type1`.
- `reactapp.py`
  - FastAPI backend exposing data and utility endpoints.
- `requirementsreact.text`
  - Python dependencies for backend runtime.

## Architecture Overview

### Frontend (React + Mapbox)

Main files in `frontend/src/`:

- `main.jsx`
  - App bootstrap and global stylesheet imports.

- `App.jsx`
  - Top-level state container and composition layer.
  - Owns selection state, confirmation workflow, chat state, and scenario state.
  - Lazy-loads `MapView` to keep the main app chunk lighter.
  - Passes data and callbacks into smaller feature components.

- `components/LeftDock.jsx`
  - Left sidebar for model info, scenario controls, saved scenarios, and simulation actions.

- `components/ChatPanel.jsx`
  - Floating chat panel for Ollama conversation and panel collapse behavior.

- `components/SelectionPanel.jsx`
  - Bottom-center selected buildings panel with confirm/reset actions and building cards.

- `components/RightPanel.jsx`
  - Right floating panel for construction-type feature definition.
  - Shows use-type and mapbox-type filtered refurbishment/detail/year-range options for area-selected confirmed buildings.

- `components/MapView.jsx`
  - Map rendering and draw interaction.
  - Performs frontend-side selection against Mapbox `building` features.
  - Resolves Mapbox building-part features by looking up parent building type via shared `building_id` field.
  - Applies mapping (`mapbox_type` -> `cea_use_type1`) and basic feature enrichment.
  - Renders confirmation state colors for selected buildings during construction phase.

- `components/common/`
  - Shared UI primitives used across features.
  - `CollapsiblePanel.jsx`: reusable shell for floating left/right bottom panels.
  - `LabeledSelectField.jsx`: reusable labeled select control for configuration forms.
  - `LeftDockTab.jsx`: reusable tab wrapper for left-dock accordion sections.

- `services/api.js`
  - API client wrappers for backend endpoints.

- `utils/selection.js`
  - Shared selection/mapbox helpers (GeoJSON parsing, stable feature keys, mapbox type normalization).

- `index.css`
  - Global layout + component styles.

For a more detailed component-by-component explanation, see [frontend/src/components/COMPONENTS_OVERVIEW.md](frontend/src/components/COMPONENTS_OVERVIEW.md).

### Backend (FastAPI)

`reactapp.py` endpoints:

- `GET /api/buildings`
  - Returns shapefile buildings as GeoJSON.
- `POST /api/select`
  - Backend selection endpoint based on drawn geometry (SHP workflow).
- `POST /api/export-cea-shp`
  - Converts selected Mapbox GeoJSON into a CEA-compatible shapefile ZIP.
  - Enriches selected buildings with reverse-geocoded address fields when available.
- `GET /api/mapbox-cea-use-type-mapping`
  - Serves CSV-based mapping for frontend selection.
- `GET /api/construction-type-mapping`
  - Serves normalized construction-type decomposition rows from CSV.
- `POST /api/chat`
  - Forwards prompts to Ollama (`/api/chat` with `/api/generate` fallback).

## Current Selection Workflow

1. User draws polygon(s) on the map.
1. Frontend reads Mapbox building features from the basemap.
1. Turf intersection computes the selected building subset.
1. **Building-part resolution**: Features with `type="building:part"` are enriched by looking up their parent building's type via the shared `building_id` field, ensuring consistent use-type classification.
1. Frontend enriches building data with normalized `height` / `min_height`, `estimated_floors`, and mapped `cea_use_type1`.
1. User can export the selection to a CEA-ready shapefile ZIP:
   - Selected buildings are converted to CEA-compatible polygons with validated attributes.
   - Address data is reverse-geocoded from Mapbox; falls back to OSM if unavailable.
   - Streets, postal codes, and name/number fields are populated from geocoding responses.
   - Generated `zone.shp` contains individual selected buildings; `site.shp` contains the merged union.
   - ZIP includes all required CEA schema columns (`name`, `floors_ag`, `floors_bg`, `height_ag`, `height_bg`, `use_type1`, `use_type2`, `use_type3`, `construction_type`, `year`, address fields).
1. User clicks Confirm selection; confirmed buildings remain orange (pending), and construction phase is enabled.
1. User draws area(s) over confirmed buildings during construction phase.
1. Right panel shows options filtered by area-selected building use types and mapbox types, including refurbishment, detail, and year range (dynamic from mapping rows).
1. User clicks Confirm features; matching `const_type` is assigned per selected building, and defined buildings turn yellow.
1. When all confirmed buildings are defined, all confirmed buildings turn green.
1. User can click Reset selection to return to initial drawing mode.

## Mapping Workflow

### Mapbox -> CEA Use Type

`mappings/MAPBOX_TO_CEA_USE_TYPE1.csv` drives category conversion and can be updated without code changes.

### CEA Construction Type Decomposition

`mappings/DE_CONSTRUCTION_TYPE_MAPPING.csv` includes:

- `const_type`
- `year_start`
- `year_end`
- `refurbishment_type`
- `detail`
- `cea_use_type1`

This file is now used by the active construction-phase assignment workflow.

## Run Instructions

### 1) Backend

From `AUP_Gui-/reactapp`:

```bash
source .venv/bin/activate
uvicorn reactapp:app --host 127.0.0.1 --port 8000 --reload
```

### 2) Frontend

From `AUP_Gui-/reactapp/frontend`:

```bash
npm install
npm run dev
```

Frontend default URL: `http://localhost:5173`

## Environment Variables

Backend (`reactapp/.env`):

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3:latest`)

Frontend (`frontend/.env` if needed):

- `VITE_API_BASE` (default: `/api`)
- `VITE_USE_MAPBOX_BUILDINGS` (`true`/`false`)
- `VITE_MAPBOX_ACCESS_TOKEN`

## Development Notes

- Frontend build succeeds cleanly.
- `MapView` is lazy-loaded, so map dependencies are split from the main app chunk.
- The Mapbox vendor chunk is still large by nature; `vite.config.js` sets `build.chunkSizeWarningLimit` to avoid noisy false-positive warnings for this expected chunk.
- If backend imports appear unresolved in editor diagnostics, ensure the Python interpreter is set to `reactapp/.venv`.
- Keep mapping CSVs as source-of-truth where possible; avoid hardcoding category logic in components.
- The UI is intentionally split into feature components so `App.jsx` stays as the orchestration layer instead of a large monolithic page file.
- The construction mapping parser is resilient to CSV header variations (BOM / key-format differences) and normalizes rows before frontend use.

## Recent Updates

### Building-Part Type Resolution (v2.0)

- **Problem**: Mapbox building-part features (geometrically subdivided portions of larger buildings) were being marked as "unknown" use type.
- **Solution**: Implemented parent-building lookup using Mapbox's `building_id` field:
  - Building parts have a numeric `building_id` property pointing to their parent building's `id`.
  - Parent buildings contain the authoritative building use type in the `type` field.
  - Both frontend (`MapView.jsx`) and backend (`reactapp.py`) now perform parent-type resolution before CEA classification.
  - Helper functions: `buildBuildingTypeLookup()` (frontend), `build_building_type_lookup_from_features()` (backend).

### CEA Export Pipeline Enhancements (v2.0)

- **Reverse Geocoding**: Selected buildings are now enriched with address data via Mapbox Reverse Geocoding API, with OSM fallback.
- **Shapefile ZIP Generation**: Exports include both `zone.shp` (individual selected buildings) and `site.shp` (merged union).
- **CEA Schema Validation**: All required columns are populated with sensible defaults:
  - Use types inferred from Mapbox building category or height heuristics.
  - Floor counts estimated from OSM data or height approximations.
  - Construction types available for post-selection refinement in the construction phase.
- **Auditability**: Raw Mapbox types are preserved in `mapbox_type_raw` for debugging and validation.
