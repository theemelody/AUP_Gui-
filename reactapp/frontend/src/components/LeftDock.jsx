import { useState } from "react";

function LeftDock({
  sidebarHidden,
  scenarioName,
  setScenarioName,
  handleSaveScenario,
  savedScenarios,
  hasSelection,
  runSimulation
}) {
  const [activeTab, setActiveTab] = useState("simulation");

  return (
    <aside className={["left-dock", sidebarHidden ? "is-hidden" : ""].filter(Boolean).join(" ")}>
      <div className="left-dock-accordion">
        <section className={["left-dock-tab", activeTab === "simulation" ? "is-active" : ""].filter(Boolean).join(" ")}>
          <button
            type="button"
            className="left-dock-tab-header"
            aria-expanded={activeTab === "simulation"}
            aria-controls="tab-panel-simulation"
            onClick={() => setActiveTab("simulation")}
          >
            Building Workspace
          </button>
          <div id="tab-panel-simulation" className="left-dock-tab-panel">
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
          </div>
        </section>

        <section className={["left-dock-tab", activeTab === "tech-tree" ? "is-active" : ""].filter(Boolean).join(" ")}>
          <button
            type="button"
            className="left-dock-tab-header"
            aria-expanded={activeTab === "tech-tree"}
            aria-controls="tab-panel-tech-tree"
            onClick={() => setActiveTab("tech-tree")}
          >
            Tech Tree Workspace
          </button>
          <div id="tab-panel-tech-tree" className="left-dock-tab-panel">
            <div className="left-dock-empty-panel">Tech tree content will appear here.</div>
          </div>
        </section>

        <section className={["left-dock-tab", activeTab === "kpi" ? "is-active" : ""].filter(Boolean).join(" ")}>
          <button
            type="button"
            className="left-dock-tab-header"
            aria-expanded={activeTab === "kpi"}
            aria-controls="tab-panel-kpi"
            onClick={() => setActiveTab("kpi")}
          >
            KPI Workspace
          </button>
          <div id="tab-panel-kpi" className="left-dock-tab-panel">
            <div className="left-dock-empty-panel">KPI content will appear here.</div>
          </div>
        </section>

        <section className={["left-dock-tab", activeTab === "secap" ? "is-active" : ""].filter(Boolean).join(" ")}>
          <button
            type="button"
            className="left-dock-tab-header"
            aria-expanded={activeTab === "secap"}
            aria-controls="tab-panel-secap"
            onClick={() => setActiveTab("secap")}
          >
            SECAP Workspace
          </button>
          <div id="tab-panel-secap" className="left-dock-tab-panel">
            <div className="left-dock-empty-panel">SECAP content will appear here.</div>
          </div>
        </section>
      </div>
    </aside>
  );
}

export default LeftDock;
