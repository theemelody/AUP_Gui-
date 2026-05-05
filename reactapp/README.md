# ReactApp Workspace

React + FastAPI prototype for map-based building selection, CEA archetype assignment, and end-to-end energy simulation, organised as a multi-page application accessed through a persistent left-dock navigation bar.

## Folder Structure

- `frontend/` — React app (Vite + TypeScript) for map interaction, selection, and UI panels.
- `mappings/`
  - `MAPBOX_TO_CEA_USE_TYPE1.csv` — maps Mapbox building categories to CEA `use_type1`.
  - `DE_CONSTRUCTION_TYPE_MAPPING.csv` — decomposed CEA construction types with `cea_use_type1`.
- `reactapp.py` — FastAPI backend: data endpoints, CEA pipeline runner, SSE streaming.
- `requirementsreact.text` — Python dependencies for backend runtime.

## Architecture Overview

### Frontend (React + Vite + TypeScript)

The app uses a multi-page architecture where each workspace is a separate page component accessed via the LeftDock tab bar. All pages stay mounted (hidden via CSS when inactive) so state is preserved when switching tabs.

#### Entry & App Shell

- `main.jsx` — App bootstrap and global stylesheet imports.
- `App.jsx` — Re-export shell; delegates to `App.tsx` so `main.jsx` needs no changes.
- `App.tsx` — Top-level orchestration layer.
  - Owns: navigation state (via `useNavigation`), sidebar visibility, scenario management, `hasSelection` flag, and simulation state.
  - Reads `ActiveSelectionInfo` from `buildingSelection` via a callback ref to avoid unnecessary re-renders.
  - Manages SSE connection (`esRef`) for simulation progress; upserts `LogEntry` objects per step.
  - Fetches the existing scenarios list and their statuses from the backend on mount.
  - Lazy-loads all four page components; keeps them all mounted and toggles `display: none` to preserve state across tab switches.

#### Navigation

- `states/navigationMachine.ts` — Reducer-based state machine.
  - `PageId` union type: `'simulation' | 'tech-tree' | 'kpi' | 'secap'`.
  - `navigationReducer` handles `NAVIGATE` actions with no-op on same-page transitions.
- `hooks/useNavigation.ts` — `useReducer` wrapper exposing `{ activePage, navigate }`.

#### Pages (`pages/`)

All pages are lazy-loaded and persistently mounted.

- `buildingSelection.tsx` — Building Workspace. Contains all map-based selection logic, construction-phase feature definition, and selection confirmation workflow. Reports active selection state up to `App.tsx` via `onActiveSelectionChange` callback.
- `techTree.tsx` — Tech Tree Workspace (placeholder).
- `kpi.tsx` — KPI Workspace (placeholder).
- `secap.tsx` — SECAP Workspace (placeholder).

#### Components (`components/`)

- `LeftDock.jsx` — Left sidebar tab bar. Accepts `activePage` and `onNavigate` props driven by the navigation state machine. Houses scenario controls: name input, save button, scenario chips with status dots (green = simulation complete, amber = inputs ready, grey = missing), run-simulation button, and a live simulation log panel.
- `ChatPanel.jsx` — Self-contained floating chat panel. Owns all chat state (`chatMessages`, `chatInput`, `chatLoading`, `chatError`) and model state (`availableModels`, `selectedModel`). Fetches Ollama models on mount and renders a dropdown when multiple are available. No chat state in parent components.
- `SelectionPanel.jsx` — Bottom-centre panel with selected buildings, confirm/reset actions, and building cards.
- `RightPanel.jsx` — Right floating panel for construction-type feature definition, showing use-type and mapbox-type filtered refurbishment/detail/year-range options.
- `MapView.jsx` — Map rendering and draw interaction. Frontend-side building selection using Mapbox `building` features, building-part parent resolution, CEA use-type mapping, and confirmed-selection colour states.
- `components/common/`
  - `CollapsiblePanel.jsx` — Reusable shell for floating left/right bottom panels.
  - `LabeledSelectField.jsx` — Reusable labelled select control.
  - `LeftDockTab.jsx` — Reusable accordion tab wrapper for the left dock.

#### Services & Utilities

- `services/api.js` — API client wrappers for all backend endpoints (including `fetchOllamaModels`, `fetchScenarioStatus`, `sendChatMessage(message, model)`).
- `utils/selection.js` — GeoJSON parsing, stable feature keys, Mapbox type normalisation.
- `vite-env.d.ts` — Vite `import.meta.env` type declarations.

#### Styles (`styles/`)

Split CSS per feature area: `layout.css`, `left-dock.css`, `chat-panel.css`, `selection-panel.css`, `construction-panel.css`, `bottom-panels.css`, `common.css`, `base.css`, `_variables.css`.

### Backend (FastAPI — `reactapp.py`)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/buildings` | Returns shapefile buildings as GeoJSON (SHP workflow only). |
| `GET` | `/api/mapbox-cea-use-type-mapping` | Serves Mapbox → CEA use-type mapping CSV for frontend selection. |
| `GET` | `/api/construction-type-mapping` | Serves normalised construction-type decomposition rows from CSV. |
| `GET` | `/api/scenarios` | Lists all scenario folders in `scenarios/`, stripping the `-scenario` suffix. |
| `GET` | `/api/scenario-status/{name}` | Returns `complete` / `ready` / `incomplete` / `missing` based on filesystem markers. |
| `POST` | `/api/save-scenario` | Saves building selection as CEA-4 compliant shapefiles in `scenarios/{name}-scenario/`. |
| `POST` | `/api/export-cea-shp` | Converts selected GeoJSON into a CEA-4 shapefile ZIP for download. |
| `POST` | `/api/select` | Backend selection endpoint based on drawn geometry (SHP workflow). |
| `POST` | `/api/chat` | Forwards prompts to Ollama with the requested model (`req.model`). |
| `GET` | `/api/ollama-models` | Returns list of locally available Ollama models. |
| `GET` | `/api/run-simulation` | SSE stream: runs the 8-step CEA pipeline for a named scenario. |

---

## CEA Simulation Pipeline

`GET /api/run-simulation?scenario_name=…` streams progress via **Server-Sent Events**. Each event is a JSON object `{ step, status, message? }`. The final event is `{ status: "complete" }` or `{ status: "failed" }`.

Steps run in order, each as a subprocess of the CEA binary at `CEA_CMD` (default: `/home/salva/micromamba/envs/cea/bin/cea`):

| Step | CEA command | What it does |
| --- | --- | --- |
| `database-helper` | `cea database-helper --databases-path DE` | Copies DE archetype/assembly/component databases into the scenario. |
| `archetypes-mapper` | `cea archetypes-mapper` | Populates all property CSVs (HVAC, envelope, schedules, supply) from `zone.shp`. |
| `surroundings-helper` | `cea surroundings-helper` | Fetches neighbouring buildings from OSM for shading. |
| `terrain-helper` | `cea terrain-helper` | Downloads `terrain.tif` from SRTM (required by radiation). |
| `weather-helper` | `cea weather-helper` | Downloads EPW weather file for the scenario location. |
| `radiation` | `cea radiation --multiprocessing true` | Computes solar radiation via DAYSIM (slow: 5–30 min). |
| `occupancy` | `cea occupancy --multiprocessing true` | Generates hourly schedule profiles per building. |
| `demand` | `cea demand --multiprocessing true` | Computes hourly energy demand → `Total_demand.csv`. |

CLI flags use bare parameter names (`--scenario`, `--databases-path`, `--multiprocessing`), not the `section:parameter` format used in config files.

---

## CEA-4 zone.shp Format

`POST /api/save-scenario` writes a single **CEA-4 compliant** `zone.shp` that merges geometry and typology (as expected by `archetypes_mapper.py` which reads all fields from `get_zone_geometry()`):

| Column | Description |
| --- | --- |
| `name` | Building ID (`B1001`, …) |
| `floors_ag` / `floors_bg` | Floors above / below ground |
| `void_deck` | Void deck floors |
| `height_ag` / `height_bg` | Heights in metres |
| `year` | Construction year |
| `const_type` | Archetype key (e.g. `SFH_I`, `MFH_F`, `NWG_11_B`) |
| `use_type1` / `use_type1r` | Primary use type and ratio |
| `use_type2` / `use_type2r` | Secondary use type and ratio |
| `use_type3` / `use_type3r` | Tertiary use type and ratio |

A separate `typology.dbf` (CEA-3 format) is no longer written.

---

## Current Selection & Simulation Workflow

1. User draws polygon(s) on the map (Building Workspace / `simulation` tab).
2. Frontend reads Mapbox `building` features from the basemap.
3. Turf intersection computes the selected building subset.
4. **Building-part resolution**: features with `type="building:part"` inherit their parent building's type via `building_id`.
5. Frontend enriches buildings with `height` / `min_height`, `estimated_floors`, and mapped `cea_use_type1`.
6. User confirms selection; confirmed buildings stay orange (pending) and the construction phase enables.
7. User draws area(s) over confirmed buildings; the right panel shows filtered refurbishment/detail/year-range options.
8. User confirms features; matching `const_type` is assigned per building. Defined buildings turn yellow; all-defined turns green.
9. User names a scenario and saves it — backend writes `zone.shp` / `site.shp` to `scenarios/{name}-scenario/`.
10. User selects a scenario chip in the left dock and clicks "Run simulation".
11. SSE events update the simulation log (icon + label per step; error detail on failure).
12. On completion, the scenario chip turns green.

---

## Scenario Management

- Saved scenarios are listed in the left dock under the Building Workspace tab with colour-coded status dots.
- On app load, `GET /api/scenarios` populates the list; `GET /api/scenario-status/{name}` fetches each status.
- Saving a new scenario via `POST /api/save-scenario` prepends it to the list immediately.
- Status is re-fetched whenever the scenario list changes.
- Folder convention: `scenarios/{name}-scenario/` on disk; display name is `{name}`.
- Status markers: `zone.shp` present → `ready`; `outputs/data/demand/Total_demand.csv` present → `complete`.

---

## Mapping Workflow

### Mapbox → CEA Use Type

`mappings/MAPBOX_TO_CEA_USE_TYPE1.csv` drives category conversion and can be updated without code changes.

### CEA Construction Type Decomposition

`mappings/DE_CONSTRUCTION_TYPE_MAPPING.csv` columns: `const_type`, `year_start`, `year_end`, `refurbishment_type`, `detail`, `cea_use_type1`. Used by the construction-phase assignment workflow in the right panel.

---

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

---

## Environment Variables

Backend (`reactapp/.env`):

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3:latest`)
- `CEA_CMD` (default: `/home/salva/micromamba/envs/cea/bin/cea`)

Frontend (`frontend/.env` if needed):

- `VITE_API_BASE` (default: `/api`)
- `VITE_USE_MAPBOX_BUILDINGS` (`true`/`false`)
- `VITE_MAPBOX_ACCESS_TOKEN`

---

## Development Notes

- `App.jsx` is a re-export shell pointing to `App.tsx`; do not add logic there.
- All pages are kept mounted with `display: none` toggled on the wrapper div — never unmount pages to preserve state.
- Navigation state lives in `states/navigationMachine.ts` (reducer) + `hooks/useNavigation.ts` (hook).
- Scenario management and simulation state live in `App.tsx` so the LeftDock controls are accessible from any page.
- Active selection state lives in `buildingSelection.tsx` and is surfaced to `App.tsx` via a callback ref (`onActiveSelectionChange`) to avoid unnecessary re-renders.
- `ChatPanel.jsx` is fully self-contained — do not hoist chat or model state into parent components.
- TypeScript is configured via `tsconfig.json` with `allowJs: true` so `.jsx` files coexist with `.tsx` without migration.
- `vite-env.d.ts` provides `import.meta.env` types for Vite.
- The Mapbox vendor chunk is large by nature; `vite.config.js` raises `build.chunkSizeWarningLimit` to suppress expected warnings.
- Keep mapping CSVs as source-of-truth; avoid hardcoding category logic in components.
- The construction mapping parser tolerates CSV header variations (BOM, key-format differences).
- CEA CLI flags use bare parameter names, not the `section:parameter` config format — `archetypes_mapper` reads from `zone.shp` directly via `get_zone_geometry()`, so all typology columns must be present in that file.
