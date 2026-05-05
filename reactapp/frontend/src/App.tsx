import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import LeftDock from './components/LeftDock.jsx';
import { useNavigation } from './hooks/useNavigation';
import { fetchScenarios, fetchScenarioStatus, saveScenarioBuildings } from './services/api.js';
import type { ActiveSelectionInfo } from './pages/buildingSelection';
import type { PageId } from './states/navigationMachine';

const BuildingSelection = lazy(() => import('./pages/buildingSelection'));
const TechTree = lazy(() => import('./pages/techTree'));
const KPI = lazy(() => import('./pages/kpi'));
const SECAP = lazy(() => import('./pages/secap'));

const PAGE_FALLBACK = (
  <div className="map-overlay" style={{ position: 'absolute', left: 12, top: 12 }}>
    Loading...
  </div>
);

type StepStatus = 'idle' | 'running' | 'done' | 'error';
type LogEntry = { step: string; status: StepStatus; messages: string[] };
type SimStatus = 'idle' | 'running' | 'done' | 'failed';
type ScenarioStatus = 'complete' | 'ready' | 'incomplete' | 'missing';

function App() {
  const { activePage, navigate } = useNavigation();
  const [sidebarHidden, setSidebarHidden] = useState(false);

  // Scenario management state
  const [scenarioName, setScenarioName] = useState('');
  const [confirmedScenarioName, setConfirmedScenarioName] = useState('');
  const [savedScenarios, setSavedScenarios] = useState<string[]>([]);
  const [hasSelection, setHasSelection] = useState(false);

  // Scenario simulation selection + status
  const [selectedScenarioForSim, setSelectedScenarioForSim] = useState('');
  const [scenarioStatuses, setScenarioStatuses] = useState<Record<string, ScenarioStatus>>({});

  // Simulation SSE state
  const [simulationLog, setSimulationLog] = useState<LogEntry[]>([]);
  const [simulationStatus, setSimulationStatus] = useState<SimStatus>('idle');
  const esRef = useRef<EventSource | null>(null);

  // Selection/polygon refs updated by BuildingSelection via callbacks
  const activeSelectionRef = useRef<ActiveSelectionInfo | null>(null);
  const drawnPolygonRef = useRef<unknown>(null);

  const handleActiveSelectionChange = useCallback(
    (info: ActiveSelectionInfo | null) => {
      activeSelectionRef.current = info;
      setHasSelection(Boolean(info?.count && info.count > 0));
    },
    [],
  );

  const handleDrawnPolygonChange = useCallback((polygon: unknown) => {
    drawnPolygonRef.current = polygon;
  }, []);

  const handleSaveScenario = useCallback(async () => {
    const name = scenarioName.trim();
    if (!name) return;
    const sel = activeSelectionRef.current;
    if (!sel?.selectedGeoJSON || !sel?.count) {
      alert('Please select buildings first before saving a scenario.');
      return;
    }
    try {
      const result = await saveScenarioBuildings(
        sel.selectedGeoJSON,
        name,
        drawnPolygonRef.current,
      );
      if (result.success) {
        setSavedScenarios((prev: string[]) =>
          prev.includes(name) ? prev : [name, ...prev].slice(0, 8),
        );
        setConfirmedScenarioName(name);
        // Refresh status for the newly saved scenario
        fetchScenarioStatus(name).then((status) =>
          setScenarioStatuses((prev) => ({ ...prev, [name]: status as ScenarioStatus })),
        );
        alert(`Scenario '${name}' saved successfully to ${result.scenario_path}`);
      }
    } catch (e) {
      alert((e as Error)?.message || 'Failed to save scenario');
    }
  }, [scenarioName]);

  const runSimulation = useCallback(() => {
    const name = selectedScenarioForSim;
    if (!name) return;

    esRef.current?.close();
    setSimulationLog([]);
    setSimulationStatus('running');

    const es = new EventSource(`/api/run-simulation?scenario_name=${encodeURIComponent(name)}`);
    esRef.current = es;

    es.onmessage = (evt) => {
      const data = JSON.parse(evt.data);

      if (data.status === 'complete' || data.status === 'failed') {
        setSimulationStatus(data.status === 'complete' ? 'done' : 'failed');
        if (data.status === 'complete') {
          setScenarioStatuses((prev) => ({ ...prev, [name]: 'complete' }));
        }
        es.close();
        return;
      }

      setSimulationLog((prev) => {
        const idx = prev.findIndex((e) => e.step === data.step);
        const entry: LogEntry =
          idx >= 0
            ? {
                ...prev[idx],
                status: data.status,
                messages: data.message
                  ? [...prev[idx].messages, data.message]
                  : prev[idx].messages,
              }
            : {
                step: data.step,
                status: data.status,
                messages: data.message ? [data.message] : [],
              };
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = entry;
          return next;
        }
        return [...prev, entry];
      });
    };

    es.onerror = () => {
      setSimulationStatus('failed');
      es.close();
    };
  }, [selectedScenarioForSim]);

  // Load saved scenarios on mount
  useEffect(() => {
    fetchScenarios()
      .then((names) => setSavedScenarios(names))
      .catch(() => {/* silently ignore if backend is not running */});
  }, []);

  // Refresh scenario statuses whenever the list changes
  useEffect(() => {
    if (!savedScenarios.length) return;
    savedScenarios.forEach((name) => {
      fetchScenarioStatus(name)
        .then((status) =>
          setScenarioStatuses((prev) => ({ ...prev, [name]: status as ScenarioStatus })),
        )
        .catch(() => {/* ignore */});
    });
  }, [savedScenarios]);

  const activeScenarioName = confirmedScenarioName || scenarioName.trim();
  const activeScenarioPath = activeScenarioName
    ? `scenarios/${activeScenarioName}-scenario`
    : '';

  return (
    <div
      className={['app-root', sidebarHidden ? 'sidebar-hidden' : '']
        .filter(Boolean)
        .join(' ')}
    >
      <LeftDock
        sidebarHidden={sidebarHidden}
        activePage={activePage}
        onNavigate={navigate as (page: PageId) => void}
        scenarioName={scenarioName}
        setScenarioName={setScenarioName}
        handleSaveScenario={handleSaveScenario}
        savedScenarios={savedScenarios}
        scenarioPath={activeScenarioPath}
        hasSelection={hasSelection}
        runSimulation={runSimulation}
        selectedScenarioForSim={selectedScenarioForSim}
        setSelectedScenarioForSim={setSelectedScenarioForSim}
        scenarioStatuses={scenarioStatuses}
        simulationLog={simulationLog}
        simulationStatus={simulationStatus}
      />

      <div className="center-column">
        <button
          type="button"
          className="sidebar-toggle-btn"
          aria-label={sidebarHidden ? 'Show sidebar' : 'Hide sidebar'}
          onClick={() => setSidebarHidden((v) => !v)}
        >
          {sidebarHidden ? '▶' : '◀'}
        </button>

        <div style={{ display: activePage === 'simulation' ? undefined : 'none' }}>
          <Suspense fallback={PAGE_FALLBACK}>
            <BuildingSelection
              onActiveSelectionChange={handleActiveSelectionChange}
              onDrawnPolygonChange={handleDrawnPolygonChange}
            />
          </Suspense>
        </div>
        <div style={{ display: activePage === 'tech-tree' ? undefined : 'none' }}>
          <Suspense fallback={PAGE_FALLBACK}>
            <TechTree />
          </Suspense>
        </div>
        <div style={{ display: activePage === 'kpi' ? undefined : 'none' }}>
          <Suspense fallback={PAGE_FALLBACK}>
            <KPI />
          </Suspense>
        </div>
        <div style={{ display: activePage === 'secap' ? undefined : 'none' }}>
          <Suspense fallback={PAGE_FALLBACK}>
            <SECAP />
          </Suspense>
        </div>
      </div>
    </div>
  );
}

export default App;
