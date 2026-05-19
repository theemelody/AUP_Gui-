import { useEffect, useState } from 'react';
import LeftDockTab from './common/LeftDockTab.jsx';
import { useScenarioContext } from '../context/ScenarioContext';
import type { SimProfile } from '../context/ScenarioContext';
import type { PageId } from '../states/navigationMachine';
import SecapDockSection from './secap/SecapDockSection.js';

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

const STEP_ICON: Record<string, string> = { idle: '○', running: '⟳', done: '✓', error: '✗' };
const STEP_LABEL: Record<string, string> = {
  'database-helper':              'Load databases',
  'archetypes-mapper':            'Map archetypes',
  'surroundings-helper':          'Fetch surroundings',
  'terrain-helper':               'Fetch terrain',
  'weather-helper':               'Fetch weather',
  'radiation':                    'Solar radiation',
  'occupancy':                    'Occupancy profiles',
  'demand':                       'Energy demand',
  'emissions':                    'Lifecycle emissions',
  'system-costs':                 'System costs',
  'photovoltaic':                 'PV potential',
  'photovoltaic-thermal':         'PVT potential',
  'solar-collector':              'Solar collector',
  'shallow-geothermal-potential': 'Geothermal potential',
  'sewage-potential':             'Sewage heat',
  'network-layout':               'Network layout',
  'thermal-network':              'Thermal network',
};

const PROFILE_OPTIONS: { value: SimProfile; label: string }[] = [
  { value: 'demand',     label: 'A — Demand Forecast' },
  { value: 'lifecycle',  label: 'B — Lifecycle Assessment' },
  { value: 'renewables', label: 'C — Renewable Potentials' },
  { value: 'network',    label: 'D — District Network' },
  { value: 'full',       label: 'E — Full Assessment' },
];

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
    scenarioStatuses, runSimulation, simulationLog, simulationStatus, simulationTotal,
    simProfile, setSimProfile,
    saveStatus, saveStartedAt, saveDuration,
  } = useScenarioContext();

  // Tick every second to refresh live durations while saving or simulating.
  const [, setTick] = useState(0);
  const needsTick = saveStatus === 'saving' || simulationStatus === 'running';
  useEffect(() => {
    if (!needsTick) return;
    const id = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [needsTick]);

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
                disabled={!scenarioName.trim() || !hasSelection || saveStatus === 'saving'}
              >
                Save
              </button>
            </div>
            {saveStatus !== 'idle' && (
              <div className="save-status-log">
                {saveStatus === 'saving' && saveStartedAt != null && (
                  <span className="save-status-saving">
                    ⟳ Saving {formatDuration(Date.now() - saveStartedAt)}
                  </span>
                )}
                {saveStatus === 'saved' && (
                  <span className="save-status-saved">
                    ✓ Saved {formatDuration(saveDuration ?? 0)}
                  </span>
                )}
                {saveStatus === 'failed' && (
                  <span className="save-status-failed">
                    ✗ Failed {formatDuration(saveDuration ?? 0)}
                  </span>
                )}
              </div>
            )}

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

            <div className="left-dock-section-title" style={{ marginTop: 8 }}>Simulation profile</div>
            <select
              className="scenario-input"
              value={simProfile}
              onChange={(e) => setSimProfile(e.target.value as SimProfile)}
              disabled={simulationStatus === 'running'}
              style={{ width: '100%', marginBottom: 8 }}
            >
              {PROFILE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>

            <button
              type="button"
              className="left-dock-run-btn"
              disabled={!selectedScenarioForSim || simulationStatus === 'running'}
              onClick={runSimulation}
            >
              {simulationStatus === 'running' ? 'Running…' : 'Run simulation'}
            </button>

            {simulationStatus !== 'idle' && simulationTotal > 0 && (() => {
              const done = simulationLog.filter((e) => e.status === 'done' || e.status === 'error').length;
              const pct = Math.round((done / simulationTotal) * 100);
              const fillClass = simulationStatus === 'done' ? 'is-done' : simulationStatus === 'failed' ? 'is-failed' : '';
              return (
                <div className="sim-progress">
                  <div className="sim-progress-track">
                    <div className={`sim-progress-fill ${fillClass}`} style={{ width: `${pct}%` }} />
                  </div>
                  <div className="sim-progress-label">
                    <span>{done} / {simulationTotal} steps</span>
                    <span>{pct}%</span>
                  </div>
                </div>
              );
            })()}
            {simulationStatus !== 'idle' && (
              <div className="simulation-log">
                {simulationLog.map((entry) => {
                  const elapsed =
                    entry.finishedAt && entry.startedAt
                      ? entry.finishedAt - entry.startedAt
                      : entry.status === 'running' && entry.startedAt
                      ? Date.now() - entry.startedAt
                      : null;
                  return (
                    <div key={entry.step} className={`sim-log-entry sim-log-${entry.status}`}>
                      <div className="sim-log-row">
                        <span className="sim-log-icon">{STEP_ICON[entry.status]}</span>
                        <span className="sim-log-label">{STEP_LABEL[entry.step] ?? entry.step}</span>
                        {elapsed != null && (
                          <span className="sim-log-time">{formatDuration(elapsed)}</span>
                        )}
                      </div>
                      {entry.status === 'error' && entry.messages.length > 0 && (
                        <div className="sim-log-detail">
                          {entry.messages[entry.messages.length - 1]}
                        </div>
                      )}
                      {entry.status === 'done' && entry.messages.some((m) => m.includes('skipped')) && (
                        <div className="sim-log-detail" style={{ opacity: 0.55 }}>skipped</div>
                      )}
                    </div>
                  );
                })}
                {simulationStatus === 'done' && (() => {
                  const total = simulationLog.reduce(
                    (acc, e) => acc + (e.startedAt && e.finishedAt ? e.finishedAt - e.startedAt : 0), 0,
                  );
                  return (
                    <div className="sim-log-complete">
                      Simulation complete {formatDuration(total)}
                    </div>
                  );
                })()}
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
          <div className="left-dock-section">
            {selectedScenarioForSim ? (
              <div className="left-dock-muted" style={{ fontSize: 11 }}>
                Viewing: <strong>{selectedScenarioForSim}</strong>
              </div>
            ) : (
              <div className="left-dock-muted">Select a scenario in Building Workspace to view KPIs.</div>
            )}
          </div>
        </LeftDockTab>

        <LeftDockTab
          id="secap"
          title="SECAP Workspace"
          active={activePage === 'secap'}
          onActivate={() => onNavigate('secap')}
        >
          <SecapDockSection />
        </LeftDockTab>

      </div>
    </aside>
  );
}

export default LeftDock;
