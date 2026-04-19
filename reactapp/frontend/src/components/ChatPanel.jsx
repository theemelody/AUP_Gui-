function ChatPanel({
  leftCollapsed,
  setLeftCollapsed,
  activeSelectionCount,
  chatMessages,
  chatLoading,
  chatError,
  chatInput,
  setChatInput,
  handleSendChat
}) {
  return (
    <aside className={["bottom-panel", "bottom-panel-left", leftCollapsed ? "is-collapsed" : ""].filter(Boolean).join(" ")}>
      <div className="bottom-panel-header">
        <div className="bottom-panel-title">
          Chat
          {activeSelectionCount > 0 ? ` - Selected: ${activeSelectionCount}` : ""}
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
  );
}

export default ChatPanel;
