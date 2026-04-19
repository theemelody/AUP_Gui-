function LeftDock({
  sidebarHidden,
  scenarioName,
  setScenarioName,
  handleSaveScenario,
  savedScenarios,
  hasSelection,
  runSimulation,
  setSidebarHidden
}) {
  return (
    <aside className={["left-dock", sidebarHidden ? "is-hidden" : ""].filter(Boolean).join(" ")}>
      <div className="left-dock-header">
        <div className="left-dock-title">Simulation Workspace</div>
        <button
          type="button"
          className="left-dock-toggle"
          aria-label="Hide sidebar"
          onClick={() => setSidebarHidden(true)}
        >
          ◀
        </button>
      </div>

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
            disabled={!scenarioName.trim()}
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
    </aside>
  );
}

export default LeftDock;
