import { lazy, Suspense, useCallback, useRef, useState } from 'react';
import LeftDock from './components/LeftDock.jsx';
import { useNavigation } from './hooks/useNavigation';
import { ScenarioProvider, useScenarioContext } from './context/ScenarioContext';
import type { ActiveSelectionInfo } from './pages/buildingSelection';
import type { PageId } from './states/navigationMachine';

const BuildingSelection = lazy(() => import('./pages/buildingSelection'));
const TechTree = lazy(() => import('./pages/techTree'));
const KPI = lazy(() => import('./pages/kpi'));
const SECAP = lazy(() => import('./pages/secap'));

const PAGE_FALLBACK = (
  <div className="map-overlay map-overlay-top-left">
    Loading...
  </div>
);

interface AppContentProps {
  activePage: PageId;
  navigate: (page: PageId) => void;
  activeSelectionRef: React.RefObject<ActiveSelectionInfo | null>;
  drawnPolygonRef: React.RefObject<unknown>;
}

function AppContent({ activePage, navigate, activeSelectionRef, drawnPolygonRef }: AppContentProps) {
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [hasSelection, setHasSelection] = useState(false);
  const [activeConstTypes, setActiveConstTypes] = useState<string[]>([]);
  const [buildingCountByConstType, setBuildingCountByConstType] = useState<Record<string, number>>({});

  const { confirmedScenarioName, loadedScenario } = useScenarioContext();

  const handleActiveSelectionChange = useCallback(
    (info: ActiveSelectionInfo | null) => {
      activeSelectionRef.current = info;
      setHasSelection(Boolean(info?.count && info.count > 0));
    },
    [activeSelectionRef],
  );

  const handleDrawnPolygonChange = useCallback(
    (polygon: unknown) => { drawnPolygonRef.current = polygon; },
    [drawnPolygonRef],
  );

  return (
    <div
      className={['app-root', sidebarHidden ? 'sidebar-hidden' : ''].filter(Boolean).join(' ')}
    >
      <LeftDock
        sidebarHidden={sidebarHidden}
        activePage={activePage}
        onNavigate={navigate}
        hasSelection={hasSelection}
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
              onConstTypesChange={setActiveConstTypes}
              onBuildingCountsChange={setBuildingCountByConstType}
              loadedScenario={loadedScenario}
            />
          </Suspense>
        </div>
        <div style={{ display: activePage === 'tech-tree' ? undefined : 'none' }}>
          <Suspense fallback={PAGE_FALLBACK}>
            <TechTree
              activeConstTypes={activeConstTypes}
              scenarioName={confirmedScenarioName}
              isActive={activePage === 'tech-tree'}
              buildingCountByConstType={buildingCountByConstType}
              totalBuildings={Object.values(buildingCountByConstType).reduce((a, b) => a + b, 0)}
            />
          </Suspense>
        </div>
        <div style={{ display: activePage === 'kpi' ? undefined : 'none' }}>
          <Suspense fallback={PAGE_FALLBACK}><KPI /></Suspense>
        </div>
        <div style={{ display: activePage === 'secap' ? undefined : 'none' }}>
          <Suspense fallback={PAGE_FALLBACK}><SECAP /></Suspense>
        </div>
      </div>
    </div>
  );
}

function App() {
  const { activePage, navigate } = useNavigation();
  const activeSelectionRef = useRef<ActiveSelectionInfo | null>(null);
  const drawnPolygonRef = useRef<unknown>(null);

  return (
    <ScenarioProvider
      navigate={navigate as (page: PageId) => void}
      getActiveSelection={() => activeSelectionRef.current}
      getDrawnPolygon={() => drawnPolygonRef.current}
    >
      <AppContent
        activePage={activePage}
        navigate={navigate as (page: PageId) => void}
        activeSelectionRef={activeSelectionRef}
        drawnPolygonRef={drawnPolygonRef}
      />
    </ScenarioProvider>
  );
}

export default App;
