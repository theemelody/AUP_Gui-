# CLAUDE.md — AUP React+FastAPI App

## Running the app

```bash
# Backend (from AUP_Gui-/reactapp/)
uvicorn reactapp:app --reload --port 8000

# Frontend (from AUP_Gui-/reactapp/frontend/)
npm run dev
```

The frontend dev server proxies `/api/*` to `http://localhost:8000` via the Vite config.

## Architecture

```
reactapp/
├── reactapp.py              # FastAPI backend — buildings, scenarios, CEA simulation SSE, KPI data
├── run_simplified_lca.py    # Workaround runner for CEA embodied LCA (see Known Bugs)
├── frontend/src/
│   ├── App.tsx               # Root: ScenarioProvider + AppContent layout
│   ├── pages/
│   │   ├── buildingSelection.tsx  # Map + selection workflow (thin wrapper, logic in hooks)
│   │   ├── techTree.tsx           # ReactFlow tech tree with ELK layout
│   │   ├── kpi.tsx                # KPI dashboard — 5 tabs, ECharts
│   │   └── secap.tsx              # SECAP page (stub)
│   ├── components/
│   │   ├── LeftDock.jsx    # Sidebar: scenario save/select/run + profile dropdown, reads ScenarioContext
│   │   ├── ChartCard.tsx   # Reusable chart card with locked state
│   │   ├── MapView.jsx     # Mapbox GL map (thin wrapper, logic in hooks)
│   │   ├── RightPanel.jsx  # Construction type assignment cascade dropdowns
│   │   ├── SelectionPanel.jsx
│   │   ├── ChatPanel.jsx
│   │   └── index.ts        # Barrel export
│   ├── hooks/
│   │   ├── useNavigation.ts       # Page navigation state machine
│   │   ├── useMapboxMapping.ts    # Loads/caches CEA use-type mapping
│   │   ├── useMapbox3D.ts         # Terrain + 3D buildings layer init
│   │   ├── useMapboxDraw.ts       # MapboxDraw lifecycle + selection/construction logic
│   │   ├── useMapboxFit.ts        # fitBounds on scenario restore
│   │   ├── useBuildingSelection.ts # 3-phase selection state machine
│   │   └── index.ts               # Barrel export
│   ├── context/
│   │   ├── ScenarioContext.tsx    # Scenario CRUD, simulation SSE, simProfile state
│   │   └── index.ts               # Barrel export
│   ├── services/api.ts    # All backend fetch wrappers (fetchKpiData, fetchScenarios, …)
│   ├── utils/
│   │   ├── kpiColors.ts   # CEA-aligned chart colors, removeZeroSeries(), FIELD_LABELS
│   │   ├── selection.js   # GeoJSON helpers, key normalization
│   │   └── mapbox.js      # Mapbox building helpers, layer definitions
│   └── styles/            # CSS design system (see _variables.css for tokens; kpi.css for KPI page)
```

## Key design decisions

**Persistent page mount**: All pages are always mounted (`display: none` toggling), not lazy-unmounted. This preserves ReactFlow and Mapbox GL state when switching tabs.

**ScenarioContext vs props**: Scenario state (save/select/run) lives in context so LeftDock doesn't need props drilled through App. The map selection state stays local in `useBuildingSelection` because it's only needed in the BuildingSelection page.

**getActiveSelection / getDrawnPolygon getters**: ScenarioProvider accepts callback getters rather than stateful values. This prevents re-renders when the map selection changes but a save isn't happening.

**MapboxDraw event handlers capture refs**: All draw event handlers use ref snapshots (`selectionLockedRef`, etc.) so stale closures never read old prop values.

**3-phase building selection**:
1. Draw polygon → `selection` (raw, unlocked)
2. "Confirm" → `confirmedSelection` (locked, orange extrusions)
3. Draw sub-area + assign type → `buildingAssignments` (per-building const_type)

## CEA simulation pipeline

The SSE endpoint `/api/run-simulation?scenario_name=X&profile=demand` streams JSON events:

```json
{"step": "database-helper", "status": "running", "message": "..."}
{"step": "database-helper", "status": "done"}          // or "done" + "skipped (already complete)"
{"status": "complete"}
```

**Step-skipping:** Every step checks its primary output for existence before running. Already-complete steps emit `status: "done"` immediately without re-executing.

**Simulation profiles** (`profile` query param):

| Profile | Extra steps beyond base demand pipeline |
|---|---|
| `demand` (default) | — |
| `lifecycle` | `final-energy`, `emissions`, `system-costs` |
| `renewables` | `photovoltaic`, `photovoltaic-thermal`, `solar-collector`, `shallow-geothermal-potential`, `sewage-potential` |
| `network` | `network-layout`, `thermal-network` |
| `full` | all of the above |

**Soft-fail steps:** All extra steps except `final-energy` are soft-fail — a non-zero exit code logs an error and continues the pipeline instead of aborting. `final-energy` is hard-fail because `emissions` and `system-costs` depend on it. Base steps (`database-helper` through `demand`) also abort on failure.

**KPI data endpoint:** `GET /api/kpi-data/{scenario_name}` reads all available CEA output CSVs and returns structured JSON consumed by the KPI dashboard.

## CEA 4.x what-if architecture (as of 2026-05-07 pull)

CEA 4.x introduced a `final-energy` step and a "what-if" namespace system. All lifecycle analysis steps now require a named what-if scenario:

```
demand → final-energy (what-if=baseline) → emissions (what-if=baseline) → system-costs (what-if=baseline)
```

Output paths (all under the scenario root):
- Final energy: `outputs/data/final-energy/baseline/final_energy_buildings.csv`
- Emissions:    `outputs/data/analysis/baseline/emissions/emissions_buildings.csv`
- Costs:        `outputs/data/analysis/baseline/costs/costs_buildings.csv`

The `configuration.yml` needed by `system-costs` is auto-created by `final-energy`.

**Column schema changes from CEA 3.x:**

| File | Old columns | New columns |
|---|---|---|
| `emissions_buildings.csv` | `GHG_sys_building_scale_tonCO2`, `GHG_sys_embodied_tonCO2yr` | `operation_kgCO2e`, `production_kgCO2e`, `type` ('building'/'plant') |
| `costs_buildings.csv` | `Capex_a_sys_building_scale_USD`, `Opex_a_sys_building_scale_USD` | `capex_a_USD`, `opex_fixed_a_USD`, `opex_var_a_USD`, `type` |

Both files include a `type` column ('building' or 'plant') for district-scale rows.

**Legacy bugs in CEA (still present but bypassed):**
- `lca_operation()` crashes with `KeyError: 'feedstock'` — only reached via legacy `emissions_simplified()`, not the what-if path
- `emissions_detailed()` crashes with `AttributeError: Parameter not found: buildings` — only reached when no `what-if-name` is set
- The `run_simplified_lca.py` workaround is no longer used (kept for reference only)

**Note: `CityEnergyAnalyst/` is upstream read-only — do not edit any files inside it.**

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VITE_MAPBOX_ACCESS_TOKEN` | — | Required: Mapbox public token |
| `VITE_USE_MAPBOX_BUILDINGS` | `true` | Use Mapbox basemap buildings (vs SHP file) |
| `CEA_CMD` | `cea` | Path to CEA CLI executable |
| `CORS_ORIGINS` | localhost list | Comma-separated allowed origins |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API for ChatPanel |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
