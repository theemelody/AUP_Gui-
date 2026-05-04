# ReactApp Workspace

React + FastAPI prototype for map-based building selection and CEA-oriented classification, organized as a multi-page application accessed through a persistent left-dock navigation bar.

## Folder Structure

- `frontend/`
  - React app (Vite + TypeScript) for map interaction, selection, and UI panels.
- `mappings/`
  - CSV mapping files used by the frontend/backend workflow.
  - `MAPBOX_TO_CEA_USE_TYPE1.csv`: maps Mapbox building categories to CEA `use_type1`.
  - `DE_CONSTRUCTION_TYPE_MAPPING.csv`: decomposed CEA construction types with `cea_use_type1`.
- `reactapp.py`
  - FastAPI backend exposing data and utility endpoints.
- `requirementsreact.text`
  - Python dependencies for backend runtime.

## Architecture Overview

### Frontend (React + Vite + TypeScript)

The app uses a multi-page architecture where each workspace is a separate page component accessed via the LeftDock tab bar. All pages stay mounted (hidden via CSS when inactive) so state is preserved when switching tabs.

#### Entry & App Shell

- `main.jsx` — App bootstrap and global stylesheet imports.
- `App.jsx` — Re-export shell; delegates to `App.tsx` so `main.jsx` needs no changes.
- `App.tsx` — Top-level orchestration layer.
  - Owns: navigation state (via `useNavigation`), sidebar visibility, scenario management state, and `hasSelection` flag.
  - Reads `ActiveSelectionInfo` from `buildingSelection` via a callback ref to avoid unnecessary re-renders.
  - Lazy-loads all four page components; keeps them all mounted and toggles `display: none` to preserve state across tab switches.
  - Fetches the existing scenarios list from the backend on mount.

#### Navigation

- `states/navigationMachine.ts` — Reducer-based state machine.
  - `PageId` union type: `'simulation' | 'tech-tree' | 'kpi' | 'secap'`.
  - `navigationReducer` handles `NAVIGATE` actions with no-op on same-page transitions.
- `hooks/useNavigation.ts` — `useReducer` wrapper exposing `{ activePage, navigate }`.

#### Pages (`pages/`)

All pages are lazy-loaded and persistently mounted.

- `buildingSelection.tsx` — Building Workspace. Contains all map-based selection logic, chat panel, construction-phase feature definition, and selection confirmation workflow. Reports active selection state up to `App.tsx` via `onActiveSelectionChange` callback.
- `techTree.tsx` — Tech Tree Workspace (placeholder).
- `kpi.tsx` — KPI Workspace (placeholder).
- `secap.tsx` — SECAP Workspace (placeholder).

#### Components (`components/`)

- `LeftDock.jsx` — Left sidebar tab bar. Accepts `activePage` and `onNavigate` props driven by the navigation state machine. Houses scenario controls (name input, save button, saved scenario chips, run simulation) which remain accessible from any page.
- `ChatPanel.jsx` — Floating chat panel for Ollama conversation and panel collapse behavior.
- `SelectionPanel.jsx` — Bottom-center panel with selected buildings, confirm/reset actions, and building cards.
- `RightPanel.jsx` — Right floating panel for construction-type feature definition, showing use-type and mapbox-type filtered refurbishment/detail/year-range options.
- `MapView.jsx` — Map rendering and draw interaction. Frontend-side building selection using Mapbox `building` features, building-part parent resolution, CEA use-type mapping, and confirmed-selection color states.
- `components/common/`
  - `CollapsiblePanel.jsx` — Reusable shell for floating left/right bottom panels.
  - `LabeledSelectField.jsx` — Reusable labeled select control.
  - `LeftDockTab.jsx` — Reusable accordion tab wrapper for the left dock.

#### Services & Utilities

- `services/api.js` — API client wrappers for all backend endpoints.
- `utils/selection.js` — GeoJSON parsing, stable feature keys, Mapbox type normalization.
- `vite-env.d.ts` — Vite `import.meta.env` type declarations.

#### Styles (`styles/`)

Split CSS modules per feature area: `layout.css`, `left-dock.css`, `chat-panel.css`, `selection-panel.css`, `construction-panel.css`, `bottom-panels.css`, `common.css`, `base.css`, `_variables.css`.

### Backend (FastAPI — `reactapp.py`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/buildings` | Returns shapefile buildings as GeoJSON (SHP workflow only). |
| `GET` | `/api/mapbox-cea-use-type-mapping` | Serves CSV-based Mapbox → CEA use-type mapping for frontend selection. |
| `GET` | `/api/construction-type-mapping` | Serves normalized construction-type decomposition rows from CSV. |
| `GET` | `/api/scenarios` | Lists all scenario folders in `scenarios/`, stripping the `-scenario` suffix. |
| `POST` | `/api/save-scenario` | Saves building selection as uncompressed shapefiles in `scenarios/{name}-scenario/`. |
| `POST` | `/api/export-cea-shp` | Converts selected Mapbox GeoJSON into a CEA-compatible shapefile ZIP (download). |
| `POST` | `/api/select` | Backend selection endpoint based on drawn geometry (SHP workflow). |
| `POST` | `/api/chat` | Forwards prompts to Ollama (`/api/chat` with `/api/generate` fallback). |

## Current Selection Workflow

1. User draws polygon(s) on the map (Building Workspace / `simulation` tab).
2. Frontend reads Mapbox `building` features from the basemap.
3. Turf intersection computes the selected building subset.
4. **Building-part resolution**: Features with `type="building:part"` inherit their parent building's type via the shared `building_id` field.
5. Frontend enriches buildings with `height` / `min_height`, `estimated_floors`, and mapped `cea_use_type1`.
6. User confirms selection; confirmed buildings stay orange (pending) and the construction phase enables.
7. User draws area(s) over confirmed buildings; the right panel shows filtered refurbishment/detail/year-range options.
8. User confirms features; matching `const_type` is assigned per building. Defined buildings turn yellow; all-defined turns green.
9. User names a scenario and saves it — backend writes `zone.shp` / `site.shp` to `scenarios/{name}-scenario/`.
10. User clicks Reset to return to drawing mode.

## Scenario Management

- Saved scenarios are listed in the left dock under the Building Workspace tab.
- On app load, `GET /api/scenarios` populates the list from the filesystem.
- Saving a new scenario via `POST /api/save-scenario` prepends it to the list immediately.
- Folder convention: `scenarios/{name}-scenario/` on disk; display name is `{name}`.

## Mapping Workflow

### Mapbox → CEA Use Type

`mappings/MAPBOX_TO_CEA_USE_TYPE1.csv` drives category conversion and can be updated without code changes.

### CEA Construction Type Decomposition

`mappings/DE_CONSTRUCTION_TYPE_MAPPING.csv` columns: `const_type`, `year_start`, `year_end`, `refurbishment_type`, `detail`, `cea_use_type1`. Used by the construction-phase assignment workflow in the right panel.

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

- `App.jsx` is a re-export shell pointing to `App.tsx`; do not add logic there.
- All pages are kept mounted with `display: none` toggled on the wrapper div — never unmount pages to preserve state.
- Navigation state lives in `states/navigationMachine.ts` (reducer) + `hooks/useNavigation.ts` (hook).
- Scenario management state lives in `App.tsx` so the LeftDock scenario controls are accessible from any page.
- Active selection state lives in `buildingSelection.tsx` and is surfaced to `App.tsx` via a callback ref (`onActiveSelectionChange`) to avoid unnecessary re-renders.
- TypeScript is configured via `tsconfig.json` with `allowJs: true` so `.jsx` files can coexist with `.tsx` files without migration.
- `vite-env.d.ts` provides `import.meta.env` types for Vite.
- The Mapbox vendor chunk is large by nature; `vite.config.js` raises `build.chunkSizeWarningLimit` to suppress expected warnings.
- Keep mapping CSVs as source-of-truth; avoid hardcoding category logic in components.
- The construction mapping parser tolerates CSV header variations (BOM, key-format differences).
