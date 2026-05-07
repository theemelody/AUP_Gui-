import LeftDockTab from './common/LeftDockTab.jsx';
import { useScenarioContext } from '../context/ScenarioContext';
import type { PageId } from '../states/navigationMachine';

const STEP_ICON: Record<string, string> = { idle: '○', running: '⟳', done: '✓', error: '✗' };
const STEP_LABEL: Record<string, string> = {
  'database-helper':     'Load databases',
  'archetypes-mapper':   'Map archetypes',
  'surroundings-helper': 'Fetch surroundings',
  'terrain-helper':      'Fetch terrain',
  'weather-helper':      'Fetch weather',
  'radiation':           'Solar radiation',
  'occupancy':           'Occupancy profiles',
  'demand':              'Energy demand',
};

interface ScenarioChipProps {
  name: string;
  status: string;
  selected: boolean;
  onSelect: (name: string) => void;
}

function ScenarioChip({ name, status, selected, onSelect }: ScenarioChipProps) {
  const dotClass =
    status === 'complete'   ? 'complete'   :
    status === 'ready'      ? 'ready'      :
    status === 'incomplete' ? 'incomplete' : 'missing';

  return (
    <div
      role="button"
      tabIndex={0}
      className={['left-dock-chip', selected ? 'is-selected' : ''].filter(Boolean).join(' ')}
      onClick={() => onSelect(selected ? '' : name)}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(selected ? '' : name); }}
    >
      <span className={`scenario-status-dot ${dotClass}`} title={status} />
      <span>{name}</span>
    </div>
  );
}

interface LeftDockProps {
  sidebarHidden: boolean;
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  hasSelection: boolean;
}

function LeftDock({ sidebarHidden, activePage, onNavigate, hasSelection }: LeftDockProps) {
  const {
    scenarioName, setScenarioName, saveScenario,
    savedScenarios, selectedScenarioForSim, selectScenarioForSim,
    scenarioStatuses, runSimulation, simulationLog, simulationStatus,
  } = useScenarioContext();

  return (
    <aside className={['left-dock', sidebarHidden ? 'is-hidden' : ''].filter(Boolean).join(' ')}>
      <div className="left-dock-accordion">

        <LeftDockTab
          id="simulation"
          title="Building Workspace"
          active={activePage === 'simulation'}
          onActivate={() => onNavigate('simulation')}
        >
          <div className="left-dock-section">
            <div className="left-dock-section-title">Simulation</div>

            <div className="scenario-save-row">
              <input
                type="text"
                className="scenario-input"
                placeholder="Scenario name"
                value={scenarioName}
                onChange={(e) => setScenarioName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') saveScenario(); }}
              />
              <button
                type="button"
                className="scenario-save-btn"
                onClick={saveScenario}
                disabled={!scenarioName.trim() || !hasSelection}
              >
                Save
              </button>
            </div>

            <div className="saved-scenarios-inline">
              {savedScenarios.length === 0 ? (
                <div className="left-dock-muted">No saved scenarios yet.</div>
              ) : (
                savedScenarios.map((name) => (
                  <ScenarioChip
                    key={name}
                    name={name}
                    status={scenarioStatuses[name] || 'missing'}
                    selected={selectedScenarioForSim === name}
                    onSelect={selectScenarioForSim}
                  />
                ))
              )}
            </div>

            <button
              type="button"
              className="left-dock-run-btn"
              disabled={!selectedScenarioForSim || simulationStatus === 'running'}
              onClick={runSimulation}
            >
              {simulationStatus === 'running' ? 'Running…' : 'Run simulation'}
            </button>

            {simulationStatus !== 'idle' && (
              <div className="simulation-log">
                {simulationLog.map((entry) => (
                  <div key={entry.step} className={`sim-log-entry sim-log-${entry.status}`}>
                    <div className="sim-log-row">
                      <span className="sim-log-icon">{STEP_ICON[entry.status]}</span>
                      <span className="sim-log-label">{STEP_LABEL[entry.step] ?? entry.step}</span>
                    </div>
                    {entry.status === 'error' && entry.messages.length > 0 && (
                      <div className="sim-log-detail">
                        {entry.messages[entry.messages.length - 1]}
                      </div>
                    )}
                  </div>
                ))}
                {simulationStatus === 'done' && (
                  <div className="sim-log-complete">Simulation complete</div>
                )}
                {simulationStatus === 'failed' && (
                  <div className="sim-log-failed">Simulation failed</div>
                )}
              </div>
            )}
          </div>
        </LeftDockTab>

        <LeftDockTab
          id="tech-tree"
          title="Tech Tree Workspace"
          active={activePage === 'tech-tree'}
          onActivate={() => onNavigate('tech-tree')}
        >
          <div className="left-dock-empty-panel">Tech tree controls will appear here.</div>
        </LeftDockTab>

        <LeftDockTab
          id="kpi"
          title="KPI Workspace"
          active={activePage === 'kpi'}
          onActivate={() => onNavigate('kpi')}
        >
          <div className="left-dock-empty-panel">KPI controls will appear here.</div>
        </LeftDockTab>

        <LeftDockTab
          id="secap"
          title="SECAP Workspace"
          active={activePage === 'secap'}
          onActivate={() => onNavigate('secap')}
        >
          <div className="left-dock-empty-panel">SECAP controls will appear here.</div>
        </LeftDockTab>

      </div>
    </aside>
  );
}

export default LeftDock;
