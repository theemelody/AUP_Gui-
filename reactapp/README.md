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
1. Frontend enriches building data with normalized `height` / `min_height`, `estimated_floors`, and mapped `cea_use_type1`.
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
