import { useCallback, useEffect, useMemo, useState } from "react";
import MapView from "./components/MapView.jsx";
import { fetchBuildings, sendChatMessage } from "./services/api.js";

function App() {
  const [buildingsGeoJSON, setBuildingsGeoJSON] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text: "Hi! I am connected to Ollama. Ask me anything about your selected buildings."
    }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);
  const [scenarioName, setScenarioName] = useState("");
  const [savedScenarios, setSavedScenarios] = useState([
    "munich-commercial-scenario"
  ]);
  const [selection, setSelection] = useState({
    count: 0,
    selectedGeoJSON: null,
    zipBase64: null,
    buildings: [],
    selectionError: null
  });

  const downloadUrl = useMemo(() => {
    if (!selection.zipBase64) return null;
    try {
      const binary = atob(selection.zipBase64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: "application/zip" });
      return URL.createObjectURL(blob);
    } catch {
      return null;
    }
  }, [selection.zipBase64]);

  useEffect(() => {
    if (!downloadUrl) return;
    return () => URL.revokeObjectURL(downloadUrl);
  }, [downloadUrl]);

  const handleSelection = useCallback((result) => {
    setSelection({
      count: result?.count || 0,
      selectedGeoJSON: result?.selected_geojson
        ? typeof result.selected_geojson === "string"
          ? JSON.parse(result.selected_geojson)
          : result.selected_geojson
        : null,
      zipBase64: result?.zip_base64 || null,
      buildings: Array.isArray(result?.buildings) ? result.buildings : [],
      selectionError: result?.selection_error || null
    });
  }, []);

  const runSimulation = useCallback(() => {
    if (!selection.selectedGeoJSON || selection.count <= 0) return;
    // Placeholder: simulation endpoint / workflow can be wired here.
    // For now, just confirm the action happened.
    alert(`Run simulation: ${selection.count} selected building(s)`);
  }, [selection.count, selection.selectedGeoJSON]);

  const handleSendChat = useCallback(async () => {
    const message = chatInput.trim();
    if (!message || chatLoading) return;

    setChatError(null);
    setChatMessages((prev) => [...prev, { role: "user", text: message }]);
    setChatInput("");
    setChatLoading(true);
    try {
      const reply = await sendChatMessage(message);
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: reply || "No response." }
      ]);
    } catch (e) {
      setChatError(e?.message || "Chat request failed");
    } finally {
      setChatLoading(false);
    }
  }, [chatInput, chatLoading]);

  const handleSaveScenario = useCallback(() => {
    const name = scenarioName.trim();
    if (!name) return;
    setSavedScenarios((prev) =>
      prev.includes(name) ? prev : [name, ...prev].slice(0, 8)
    );
    setScenarioName("");
  }, [scenarioName]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const geojson = await fetchBuildings();
        if (!cancelled) setBuildingsGeoJSON(geojson);
      } catch (e) {
        if (!cancelled)
          setLoadError(e?.message || "Failed to load buildings GeoJSON");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div
      className={[
        "app-root",
        sidebarHidden ? "sidebar-hidden" : "",
        leftCollapsed ? "left-collapsed" : "",
        rightCollapsed ? "right-collapsed" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
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
            disabled={!selection.selectedGeoJSON || selection.count <= 0}
            onClick={runSimulation}
          >
            Run simulation
          </button>
        </div>

      </aside>

      <div className="center-column">
        {sidebarHidden && (
          <button
            type="button"
            className="sidebar-reopen"
            aria-label="Show sidebar"
            onClick={() => setSidebarHidden(false)}
          >
            ▶
          </button>
        )}
        <MapView
          buildingsGeoJSON={buildingsGeoJSON}
          selectedGeoJSON={selection.selectedGeoJSON}
          onSelection={handleSelection}
        />
        {loadError && (
          <div
            className="map-overlay map-overlay-error"
            style={{
              position: "absolute",
              left: 12,
              bottom: 12,
              padding: "8px 10px",
              borderRadius: 8,
              background: "rgba(0,0,0,0.75)",
              color: "white",
              fontSize: 12,
              maxWidth: 420
            }}
          >
            Buildings load error: {loadError}
          </div>
        )}
        <aside className={["bottom-panel", "bottom-panel-left", leftCollapsed ? "is-collapsed" : ""].filter(Boolean).join(" ")}>
          <div className="bottom-panel-header">
            <div className="bottom-panel-title">
              Chat
              {selection.count > 0 ? ` — Selected: ${selection.count}` : ""}
            </div>
            <button
              type="button"
              className="bottom-panel-toggle"
              aria-label={leftCollapsed ? "Expand left panel" : "Collapse left panel"}
              onClick={() => setLeftCollapsed((v) => !v)}
            >
              {leftCollapsed ? "▲" : "▼"}
            </button>
          </div>
          <div className="bottom-panel-body">
            <div className="chatbot-section">
              <div className="left-dock-section-title">Chatbot (Ollama)</div>
              <div className="chat-window" aria-live="polite">
                {chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={[
                      "chat-message",
                      msg.role === "user" ? "chat-user" : "chat-assistant"
                    ].join(" ")}
                  >
                    <div className="chat-role">
                      {msg.role === "user" ? "You" : "Assistant"}
                    </div>
                    <div className="chat-text">{msg.text}</div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="chat-message chat-assistant">
                    <div className="chat-role">Assistant</div>
                    <div className="chat-text">Thinking...</div>
                  </div>
                )}
              </div>
              {chatError && <div className="chat-error">{chatError}</div>}
              <div className="chat-input-row">
                <input
                  type="text"
                  className="chat-input"
                  placeholder="Type your message..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSendChat();
                  }}
                />
                <button
                  type="button"
                  className="chat-send-btn"
                  onClick={handleSendChat}
                  disabled={chatLoading || !chatInput.trim()}
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </aside>

        <section className="selected-list-panel" aria-label="Selected buildings">
          <div className="selected-list-header">
            <div className="selected-list-title">
              Selected Buildings ({selection.count})
            </div>
            <div className="selected-list-actions">
              <a
                className={[
                  "action-link",
                  downloadUrl && selection.count > 0 ? "" : "is-disabled"
                ]
                  .filter(Boolean)
                  .join(" ")}
                href={downloadUrl || "#"}
                download="selected_buildings.zip"
                onClick={(e) => {
                  if (!downloadUrl || selection.count <= 0) e.preventDefault();
                }}
              >
                Download ZIP
              </a>
            </div>
          </div>

          <div className="selected-list-body">
            {selection.selectionError && (
              <div className="selected-empty selected-error">
                Selection error: {selection.selectionError}
              </div>
            )}
            {selection.buildings.length === 0 ? (
              <div className="selected-empty">
                No buildings selected yet.
              </div>
            ) : (
              selection.buildings.map((building, index) => (
                <article className="building-card" key={index}>
                  <div className="building-card-title">Building {index + 1}</div>
                  <div className="building-props">
                    {Object.entries(building).map(([key, value]) => (
                      <div className="building-prop-row" key={key}>
                        <span className="building-prop-key">{key}</span>
                        <span className="building-prop-value">
                          {value === null || value === undefined
                            ? "-"
                            : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </article>
              ))
            )}
          </div>
        </section>

        <aside className={["bottom-panel", "bottom-panel-right", rightCollapsed ? "is-collapsed" : ""].filter(Boolean).join(" ")}>
          <div className="bottom-panel-header">
            <div className="bottom-panel-title">Right panel</div>
            <button
              type="button"
              className="bottom-panel-toggle"
              aria-label={rightCollapsed ? "Expand right panel" : "Collapse right panel"}
              onClick={() => setRightCollapsed((v) => !v)}
            >
              {rightCollapsed ? "▲" : "▼"}
            </button>
          </div>
          <div className="bottom-panel-body">Content</div>
        </aside>
      </div>
    </div>
  );
}

export default App;

