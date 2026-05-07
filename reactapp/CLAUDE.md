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
├── reactapp.py          # FastAPI backend — buildings, scenarios, CEA simulation SSE
├── frontend/src/
│   ├── App.tsx           # Root: ScenarioProvider + AppContent layout
│   ├── pages/
│   │   ├── buildingSelection.tsx  # Map + selection workflow (thin wrapper, logic in hooks)
│   │   ├── techTree.tsx           # ReactFlow tech tree with ELK layout
│   │   ├── kpi.tsx                # KPI page (stub)
│   │   └── secap.tsx              # SECAP page (stub)
│   ├── components/
│   │   ├── LeftDock.jsx    # Sidebar: scenario save/select/run, reads ScenarioContext
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
│   │   ├── ScenarioContext.tsx    # Scenario CRUD, simulation SSE, loaded scenario
│   │   └── index.ts               # Barrel export
│   ├── services/api.js    # All backend fetch wrappers
│   ├── utils/
│   │   ├── selection.js   # GeoJSON helpers, key normalization
│   │   └── mapbox.js      # Mapbox building helpers, layer definitions
│   └── styles/            # CSS design system (see _variables.css for tokens)
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

The SSE endpoint `/api/run-simulation?scenario_name=X` streams JSON events:

```json
{"step": "database-helper", "status": "running", "message": "..."}
{"step": "database-helper", "status": "done"}
{"status": "complete"}
```

Steps in order: `database-helper`, `archetypes-mapper`, `surroundings-helper`,
`terrain-helper`, `weather-helper`, `radiation`, `occupancy`, `demand`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VITE_MAPBOX_ACCESS_TOKEN` | — | Required: Mapbox public token |
| `VITE_USE_MAPBOX_BUILDINGS` | `true` | Use Mapbox basemap buildings (vs SHP file) |
| `CEA_CMD` | `cea` | Path to CEA CLI executable |
| `CORS_ORIGINS` | localhost list | Comma-separated allowed origins |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API for ChatPanel |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
