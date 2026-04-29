import { useState } from "react";
import LeftDockTab from "./common/LeftDockTab.jsx";

function LeftDock({
  sidebarHidden,
  scenarioName,
  setScenarioName,
  handleSaveScenario,
  savedScenarios,
  scenarioPath,
  hasSelection,
  runSimulation
}) {
  const [activeTab, setActiveTab] = useState("simulation");
  const emptyTabs = [
    { id: "tech-tree", title: "Tech Tree Workspace", message: "Tech tree content will appear here." },
    { id: "kpi", title: "KPI Workspace", message: "KPI content will appear here." },
    { id: "secap", title: "SECAP Workspace", message: "SECAP content will appear here." }
  ];

  return (
    <aside className={["left-dock", sidebarHidden ? "is-hidden" : ""].filter(Boolean).join(" ")}>
      <div className="left-dock-accordion">
        <LeftDockTab
          id="simulation"
          title="Building Workspace"
          active={activeTab === "simulation"}
          onActivate={() => setActiveTab("simulation")}
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
        </LeftDockTab>

        {emptyTabs.map((tab) => (
          <LeftDockTab
            key={tab.id}
            id={tab.id}
            title={tab.title}
            active={activeTab === tab.id}
            onActivate={() => setActiveTab(tab.id)}
          >
            <div className="left-dock-empty-panel">{tab.message}</div>
          </LeftDockTab>
        ))}
      </div>
    </aside>
  );
}

export default LeftDock;
