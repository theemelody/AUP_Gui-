import CollapsiblePanel from "./common/CollapsiblePanel.jsx";

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
    <CollapsiblePanel
      positionClass="bottom-panel-left"
      collapsed={leftCollapsed}
      setCollapsed={setLeftCollapsed}
      title="Chat"
      titleSuffix={activeSelectionCount > 0 ? `Selected: ${activeSelectionCount}` : ""}
      expandAriaLabel="Expand left panel"
      collapseAriaLabel="Collapse left panel"
    >
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
    </CollapsiblePanel>
  );
}

export default ChatPanel;
