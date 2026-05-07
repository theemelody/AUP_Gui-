# ReactApp Workspace

React + FastAPI prototype for map-based building selection, CEA archetype assignment, end-to-end energy simulation, and interactive technology tree visualisation — organised as a multi-page application accessed through a persistent left-dock navigation bar.

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
- `App.tsx` — Top-level layout: wraps `AppContent` in `ScenarioProvider`, passes `getActiveSelection` / `getDrawnPolygon` getter callbacks so the context can read map state at save-time without triggering re-renders. Lazy-loads all four page components; keeps them all mounted and toggles `display: none` to preserve state across tab switches.

#### Navigation

- `states/navigationMachine.ts` — Reducer-based state machine.
  - `PageId` union type: `'simulation' | 'tech-tree' | 'kpi' | 'secap'`.
  - `navigationReducer` handles `NAVIGATE` actions with no-op on same-page transitions.
- `hooks/useNavigation.ts` — `useReducer` wrapper exposing `{ activePage, navigate }`.

#### Context (`context/`)

- `ScenarioContext.tsx` — Holds all scenario state: name, saved list, statuses, simulation log/status, loaded scenario. Provides `saveScenario()`, `selectScenarioForSim()`, and `runSimulation()` actions. `LeftDock` reads directly from this context — no prop drilling.

#### Hooks (`hooks/`)

- `useNavigation.ts` — Navigation state machine wrapper.
- `useBuildingSelection.ts` — Encapsulates the 3-phase building selection state machine (draw → confirm → assign construction type) including all derived state and the scenario-restore effect.
- `useMapboxMapping.ts` — Lazy-loads and caches the Mapbox → CEA use-type mapping; warms the cache on mount to avoid a race on first selection.
- `useMapbox3D.ts` — Initialises terrain DEM, sky atmosphere, and 3D basemap buildings layer on first map load.
- `useMapboxDraw.ts` — Full `MapboxDraw` lifecycle: initialisation, `fireSelection` debounced callback, phase-aware routing between initial selection and construction-area sub-selection, draw button visibility, stale shape cleanup.
- `useMapboxFit.ts` — Exposes `fitToBounds` utility; owns the `fitToGeoJSON` fly-to effect for scenario restore.

#### Pages (`pages/`)

All pages are lazy-loaded and persistently mounted.

- `buildingSelection.tsx` — Building Workspace. Thin layout wrapper: loads buildings GeoJSON and construction mapping, delegates all state logic to `useBuildingSelection`, renders MapView + panels. Reports active selection state up to `App.tsx` via `onActiveSelectionChange` callback.
- `techTree.tsx` — Tech Tree Workspace. Renders the CEA construction-type dependency graph using **ReactFlow** and the **ELK** hierarchical layout engine.
  - Fetches graph data from `/api/techtree-graph`; computes the set of active nodes by propagating `activeConstTypes` through graph edges.
  - Only active nodes are passed to ELK (typically 5–25 in a collapsed view, 50–100 expanded), avoiding layout-engine overload.
  - Collapsed sublayers (L2: Envelope / HVAC / Supply; L3: Conversion / Feedstocks) render as interactive pills showing the active node count; clicking a pill expands its nodes.
  - Layer bands (L0 Typology / L1 Archetypes / L2 Assemblies / L3 Energy) are rendered as viewport-tracked coloured overlays using `useViewport()` from `@xyflow/react`, correctly following pan and zoom.
  - ReactFlow is only mounted when the tab is active (`isActive` prop) to avoid the 0-size container warning.
- `kpi.tsx` — KPI Workspace. Full dashboard with five tab-switched views, each driven by `GET /api/kpi-data/{scenario_name}`:
  - **Demand** — 8 charts: annual end-use stacked bar, EUI per building with NZEB 200 benchmark line, peak load grouped bar, monthly demand bar, 3 load-duration curve panels (heating / cooling / electricity), hourly load curve stacked bar + operative temperature line, monthly thermal balance (gains/losses), monthly solar radiation by surface orientation.
  - **SECAP** — SECAP compliance KPI cards and charts derived from demand outputs.
  - **Emissions & Costs** — Annual GHG tCO₂-eq and CAPEX/OPEX stacked bars (locked until Profile B is run; `ChartCard locked` state shows padlock overlay).
  - **Tech Potentials** — PV vs total demand, PVT + SC thermal bar, renewable-mix donut, geothermal capacity, sewage heat (locked until Profile C).
  - **Thermal Networks** — Network demand-duration curve and per-pipe metrics (locked until Profile D).
  - Tab buttons show a lock icon when the required simulation profile has not been run; clicking a locked tab does nothing.
- `secap.tsx` — SECAP Workspace (placeholder).

#### Components (`components/`)

- `LeftDock.tsx` — Left sidebar tab bar. Reads all scenario state from `ScenarioContext` (4 props only: `sidebarHidden`, `activePage`, `onNavigate`, `hasSelection`). Houses: simulation profile selector (A–E dropdown), scenario name input, save button with live save-duration display, scenario chips with status dots, run-simulation button, and a live simulation log panel with per-step HH:MM:SS elapsed timers. Clicking a chip loads the full saved state and navigates to Building Workspace. Profile labels: A — Demand Forecast, B — Lifecycle Assessment, C — Renewable Potentials, D — District Network, E — Full Assessment.
- `ChartCard.tsx` — Reusable chart card wrapper. Props: `title`, `subtitle?`, `height?`, `locked?`, `lockMessage?`, `wide?`. When `locked=true` renders a padlock SVG overlay at the chart height instead of children. Used by all KPI tab components.
- `ChatPanel.tsx` — Self-contained floating chat panel. Owns all chat state and Ollama model state. No chat state in parent components.
- `SelectionPanel.tsx` — Bottom-centre panel with selected buildings, confirm/reset actions, and building cards.
- `RightPanel.tsx` — Right floating panel for construction-type feature definition. Cascade dropdowns (refurbishment → detail → year range) unified into a single `handleCascadeChange(useType, mapboxType, field, value)` handler.
- `MapView.tsx` — Thin map shell (~160 lines). Delegates to four focused hooks; renders `Source`/`Layer` pairs for building states. Accepts `fitToGeoJSON` for animated scenario restore.
- `components/common/`
  - `LabeledSelectField.tsx` — Reusable labelled select control.
  - `LeftDockTab.tsx` — Reusable accordion tab wrapper for the left dock.

#### Services & Utilities

- `services/api.ts` — Typed API client wrappers for all backend endpoints. Exports `fetchKpiData(scenarioName)` → `Promise<KpiData>`, with `KpiData` and `KpiAnnualRow` interfaces covering all fields returned by `/api/kpi-data/`.
- `utils/kpiColors.ts` — `KPI_COLORS` map (heating/cooling/electricity/solar/GHG/cost + orientation shades + thermal-balance keys); `FIELD_LABELS` map (CEA column names → human-readable strings); `removeZeroSeries()` helper that strips series whose absolute sum is below 1e-4.
- `utils/selection.ts` — GeoJSON parsing, stable feature keys, Mapbox type normalisation.
- `utils/mapbox.js` — Mapbox building helpers (type extraction, feature normalisation, selection logic) and layer definitions (fill, outline, extrusion).
- `vite-env.d.ts` — Vite `import.meta.env` types + ambient declarations for `@mapbox/mapbox-gl-draw` and `react-collapse`.

#### Styles (`styles/`)

Token-driven CSS design system — all colors, spacing, radii, and transitions live in `_variables.css` as CSS custom properties. Each file covers one feature area.

| File | Purpose |
| --- | --- |
| `_variables.css` | All design tokens (colors, spacing, radii, blur, transitions, button variants) |
| `base.css` | HTML/body reset, global form element styles |
| `layout.css` | App grid, map container, sidebar toggle |
| `left-dock.css` | Left sidebar header, tabs, scenario chips, simulation log, profile selector |
| `chat-panel.css` | Floating chat window, messages, input bar |
| `bottom-panels.css` | Floating panel container shell (selection + construction) |
| `selection-panel.css` | Selected buildings list and building cards |
| `construction-panel.css` | Construction phase UI, cascade dropdowns |
| `common.css` | Shared button/input base classes, status indicators, utility classes |
| `techtree.css` | Tech Tree ReactFlow canvas, layer bands, node pills |
| `kpi.css` | KPI page layout, tab bar, chart-card grid, locked state overlay |

The app has two visual zones with distinct aesthetics:

- **Dark zone** — left dock, tech tree background: navy/slate bg, light text, sky-blue accent.
- **Light zone** — map-overlay panels (chat, right, selection): frosted glass, white cards, dark text.

Button tokens (`--btn-dark-*`, `--btn-light-*`, `--btn-primary-*`, `--btn-success-*`, `--btn-danger-*`) and backdrop-blur/panel utilities in `common.css` keep the two zones consistent without mixing styles between files. See [`styles/README.md`](frontend/src/styles/README.md) for the full token catalogue and naming conventions.

### Backend (FastAPI — `reactapp.py`)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/buildings` | Returns shapefile buildings as GeoJSON (SHP workflow only). |
| `GET` | `/api/mapbox-cea-use-type-mapping` | Serves Mapbox → CEA use-type mapping CSV for frontend selection. |
| `GET` | `/api/construction-type-mapping` | Serves normalised construction-type decomposition rows from CSV. |
| `GET` | `/api/scenarios` | Lists all scenario folders in `scenarios/`, stripping the `-scenario` suffix. |
| `GET` | `/api/scenario-status/{name}` | Returns `complete` / `ready` / `incomplete` / `missing` based on filesystem markers. |
| `GET` | `/api/scenario-data/{name}` | Returns the `scenario.json` snapshot (GeoJSON + drawn polygon) for full client state restore. |
| `POST` | `/api/save-scenario` | Saves building selection as CEA-4 compliant shapefiles + `scenario.json` snapshot in `scenarios/{name}-scenario/`. |
| `POST` | `/api/export-cea-shp` | Converts selected GeoJSON into a CEA-4 shapefile ZIP for download. |
| `POST` | `/api/select` | Backend selection endpoint based on drawn geometry (SHP workflow only; dead if `USE_MAPBOX_BUILDINGS=true`). |
| `POST` | `/api/chat` | Forwards prompts to Ollama with the requested model (`req.model`). |
| `GET` | `/api/ollama-models` | Returns list of locally available Ollama models. |
| `GET` | `/api/run-simulation` | SSE stream: runs the CEA pipeline for a named scenario. Accepts `?profile=demand\|lifecycle\|renewables\|network\|full` (default `demand`). |
| `GET` | `/api/kpi-data/{name}` | Returns aggregated KPI data from CEA output CSVs: `annual` (per-building totals), `monthly` (12-month demand), `monthly_balance` (thermal gains/losses), `load_duration` (sorted hourly curves), `hourly_sample` (every 6th hour), `solar_radiation` (by surface orientation), `emissions`, `costs`, `potentials`, `network`. Each key is `null` if the required output file doesn't exist; `available[]` lists which data groups are present. |
| `GET` | `/api/techtree-graph` | Returns the CEA technology graph with per-building linkage: each node carries `buildings[]` (list of building names that use it). Accepts `?scenario_name=&region=DE`. |

---

## CEA Simulation Profiles

`GET /api/run-simulation?scenario_name=…&profile=…` streams progress via **Server-Sent Events**. Each event is a JSON object `{ step, status, message? }`. The final event is `{ status: "complete" }` or `{ status: "failed" }`. The `profile` parameter controls which steps run:

| Profile | Steps |
| --- | --- |
| `demand` (A — default) | database-helper → archetypes-mapper → surroundings-helper → terrain-helper → weather-helper → radiation → occupancy → demand |
| `lifecycle` (B) | Profile A + emissions + system-costs |
| `renewables` (C) | Profile A's radiation + photovoltaic + photovoltaic-thermal + solar-collector + shallow-geothermal-potential + sewage-potential |
| `network` (D) | Profile A + network-layout + thermal-network |
| `full` (E) | All steps from A+B+C+D |

All steps run as subprocesses of the CEA binary at `CEA_CMD` (default: `cea`; override via env var). CLI flags use bare parameter names, not `section:parameter` config format.

### Per-step reference (Profile A)

| Step | CEA command | What it does |
| --- | --- | --- |
| `database-helper` | `cea database-helper --databases-path DE` | Copies DE archetype/assembly/component databases into the scenario. |
| `archetypes-mapper` | `cea archetypes-mapper` | Populates all property CSVs (HVAC, envelope, schedules, supply) from `zone.shp`. |
| `surroundings-helper` | `cea surroundings-helper` | Fetches neighbouring buildings from OSM for shading. |
| `terrain-helper` | `cea terrain-helper` | Downloads `terrain.tif` from SRTM (required by radiation). |
| `weather-helper` | `cea weather-helper` | Downloads EPW weather file for the scenario location. |
| `radiation` | `cea radiation --multiprocessing true` | Computes solar radiation via DAYSIM (slow: 5–30 min). |
| `occupancy` | `cea occupancy --multiprocessing true` | Generates hourly schedule profiles per building. |
| `demand` | `cea demand --multiprocessing true` | Computes hourly energy demand → `Total_demand.csv` + `Total_demand_hourly.csv`. |

### Simplified LCA helper

`run_simplified_lca.py` is a standalone script for Profile B that runs `lca_embodied()` directly via the CEA Python API (bypasses the broken `emissions_simplified` config path). Run as: `python run_simplified_lca.py /path/to/scenario`. Writes embodied LCA results plus a stub operational CSV so the KPI endpoint can detect that the emissions step has been attempted.

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

- Saved scenarios are listed in the left dock with colour-coded status dots.
- On app load, `GET /api/scenarios` populates the list; `GET /api/scenario-status/{name}` fetches each status.
- Saving a new scenario via `POST /api/save-scenario` prepends it to the list immediately and writes:
  - `inputs/building-geometry/zone.shp` + `site.shp` — CEA-4 compliant geometry
  - `scenario.json` — full client state snapshot (`selected_geojson` with all assignment properties, `drawn_polygon`, timestamp)
- **Scenario restore**: clicking a chip calls `GET /api/scenario-data/{name}` to fetch `scenario.json`, then restores `confirmedSelection`, `buildingAssignments`, and `drawnPolygon` in the frontend. The map animates to fit the selection bounds; the Tech Tree highlights the active construction types. Scenarios without `scenario.json` (saved before this feature) still work for simulation — the restore step fails silently.
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

Backend (`reactapp/.env` — see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CEA_CMD` | `cea` | Path to CEA CLI executable |
| `CORS_ORIGINS` | localhost list | Comma-separated allowed origins |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base |
| `OLLAMA_MODEL` | `llama3` | Default Ollama model |

Frontend (`frontend/.env` if needed):

- `VITE_API_BASE` (default: `/api`)
- `VITE_USE_MAPBOX_BUILDINGS` (`true`/`false`)
- `VITE_MAPBOX_ACCESS_TOKEN`

---

## Development Notes

- All pages are kept mounted with `display: none` toggled on the wrapper div — never unmount pages to preserve state.
- Navigation state lives in `states/navigationMachine.ts` (reducer) + `hooks/useNavigation.ts` (hook).
- Scenario state lives in `context/ScenarioContext.tsx` so `LeftDock` can access it without prop drilling from `App.tsx`.
- Active selection state lives in `useBuildingSelection` and is surfaced to `App.tsx` via a callback ref (`onActiveSelectionChange`) to avoid unnecessary re-renders.
- `ScenarioProvider` receives `getActiveSelection` and `getDrawnPolygon` as getter callbacks rather than stateful values — this means `saveScenario()` reads the current selection at call time without causing re-renders on every map interaction.
- Derived state (`lockedSelectionGeoJSONWithState`, `confirmedBuildingsWithAssignments`) must stay `useMemo`-ised — inline object creation on every render caused an infinite re-render loop via the `onConstTypesChange` callback chain.
- `ChatPanel.tsx` is fully self-contained — do not hoist chat or model state into parent components.
- TypeScript is configured with `moduleResolution: "bundler"`. All source files are `.ts`/`.tsx` except `main.jsx` (entry point) and `utils/mapbox.js` (Mapbox layer definitions — kept `.js` to avoid layer type gymnastics).
- `vite-env.d.ts` has ambient declarations for `@mapbox/mapbox-gl-draw` and `react-collapse` (neither ships types).
- The Mapbox vendor chunk is large by nature; `vite.config.ts` raises `build.chunkSizeWarningLimit` to suppress expected warnings.
- Keep mapping CSVs as source-of-truth; avoid hardcoding category logic in components.
- CEA CLI flags use bare parameter names, not the `section:parameter` config format — `archetypes_mapper` reads from `zone.shp` directly via `get_zone_geometry()`, so all typology columns must be present in that file.
- Tech Tree must only pass **active** nodes to ELK — passing all ~1 000 CEA nodes freezes the layout engine. Filter in `buildVisibleGraph` before calling `runElkLayout`.
- `LayerOverlays` in `techTree.tsx` must be a child of `<ReactFlow>` (not a sibling) to access the `useViewport()` context for correct canvas-to-screen coordinate mapping.
- `prepare_scenario_structure()` in `reactapp.py` applies three geometry repairs before writing `zone.shp`: (1) MultiPolygon → largest polygon; (2) `buffer(0)` repair for self-intersecting rings; (3) drop buildings with projected footprint < 10 m² (CEA demand divides by floor area and NaN-crashes on near-zero footprints).
- KPI charts use `echarts-for-react` + ECharts 6. Always set a fixed `height` style on `ReactECharts` — the container must have a non-zero height or ECharts renders blank. Use `removeZeroSeries()` before passing `series` to avoid rendering ghost legend entries for unused energy carriers.
- The `ScenarioContext` exposes `simProfile` / `setSimProfile` for profile selection and `saveStatus` / `saveStartedAt` / `saveDuration` for save-feedback UX. `LeftDock` reads these directly — no prop drilling needed.
- `alert()` on successful save has been removed; the save button now shows a live HH:MM:SS counter while saving and switches to a "Saved" state on completion.
