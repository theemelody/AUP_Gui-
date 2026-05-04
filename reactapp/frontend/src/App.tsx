import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import LeftDock from './components/LeftDock.jsx';
import { useNavigation } from './hooks/useNavigation';
import { fetchScenarios, saveScenarioBuildings } from './services/api.js';
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

function App() {
  const { activePage, navigate } = useNavigation();
  const [sidebarHidden, setSidebarHidden] = useState(false);

  // Scenario management state — kept here so LeftDock can drive it regardless of active page.
  const [scenarioName, setScenarioName] = useState('');
  const [confirmedScenarioName, setConfirmedScenarioName] = useState('');
  const [savedScenarios, setSavedScenarios] = useState<string[]>([]);
  const [hasSelection, setHasSelection] = useState(false);

  // Selection/polygon refs updated by BuildingSelection via callbacks — avoids re-renders in App.
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
        alert(`Scenario '${name}' saved successfully to ${result.scenario_path}`);
      }
    } catch (e) {
      alert((e as Error)?.message || 'Failed to save scenario');
    }
  }, [scenarioName]);

  const runSimulation = useCallback(() => {
    const sel = activeSelectionRef.current;
    if (!sel?.selectedGeoJSON || !sel?.count) return;
    alert(`Run simulation: ${sel.count} selected building(s)`);
  }, []);

  useEffect(() => {
    fetchScenarios()
      .then((names) => setSavedScenarios(names))
      .catch(() => {/* silently ignore if backend is not running */});
  }, []);

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
