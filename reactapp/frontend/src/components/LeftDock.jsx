import LeftDockTab from "./common/LeftDockTab.jsx";

function LeftDock({
  sidebarHidden,
  activePage,
  onNavigate,
  scenarioName,
  setScenarioName,
  handleSaveScenario,
  savedScenarios,
  scenarioPath,
  hasSelection,
  runSimulation
}) {
  return (
    <aside className={["left-dock", sidebarHidden ? "is-hidden" : ""].filter(Boolean).join(" ")}>
      <div className="left-dock-accordion">

        <LeftDockTab
          id="simulation"
          title="Building Workspace"
          active={activePage === "simulation"}
          onActivate={() => onNavigate("simulation")}
        >
          <div className="left-dock-section">
            <div className="left-dock-section-title">Model</div>
            <div className="left-dock-field">llama3.1:8b</div>
            <div className="left-dock-status-row">
              <span className="status-dot status-off" />
              OpenAI (no key)
            </div>
            <div className="left-dock-status-row">
              <span className="status-dot status-on" />
              Ollama
            </div>
          </div>

          <div className="left-dock-section">
            <div className="left-dock-section-title">Simulation Settings</div>
            <div className="left-dock-field">/home/user/automatic-urban-planner</div>
            {scenarioPath ? <div className="left-dock-muted">{scenarioPath}</div> : null}
          </div>

          <div className="left-dock-section">
            <div className="left-dock-section-title">Simulation</div>
            <div className="scenario-save-row">
              <input
                type="text"
                className="scenario-input"
                placeholder="Scenario name"
                value={scenarioName}
                onChange={(e) => setScenarioName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveScenario();
                }}
              />
              <button
                type="button"
                className="scenario-save-btn"
                onClick={handleSaveScenario}
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
                  <div className="left-dock-chip" key={name}>
                    {name}
                  </div>
                ))
              )}
            </div>

            <button
              type="button"
              className="left-dock-run-btn"
              disabled={!hasSelection}
              onClick={runSimulation}
            >
              Run simulation
            </button>
          </div>
        </LeftDockTab>

        <LeftDockTab
          id="tech-tree"
          title="Tech Tree Workspace"
          active={activePage === "tech-tree"}
          onActivate={() => onNavigate("tech-tree")}
        >
          <div className="left-dock-empty-panel">Tech tree controls will appear here.</div>
        </LeftDockTab>

        <LeftDockTab
          id="kpi"
          title="KPI Workspace"
          active={activePage === "kpi"}
          onActivate={() => onNavigate("kpi")}
        >
          <div className="left-dock-empty-panel">KPI controls will appear here.</div>
        </LeftDockTab>

        <LeftDockTab
          id="secap"
          title="SECAP Workspace"
          active={activePage === "secap"}
          onActivate={() => onNavigate("secap")}
        >
          <div className="left-dock-empty-panel">SECAP controls will appear here.</div>
        </LeftDockTab>

      </div>
    </aside>
  );
}

export default LeftDock;
