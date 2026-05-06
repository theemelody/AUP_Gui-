import { useCallback, useEffect, useRef, useState } from "react";
import { Collapse } from "react-collapse";
import { fetchOllamaModels, sendChatMessage } from "../services/api.js";

const INITIAL_MESSAGE = {
  role: "assistant",
  text: "Hi! I am connected to Ollama. Ask me anything about your selected buildings.",
};

function ChatPanel({
  leftCollapsed,
  setLeftCollapsed,
  activeSelectionCount,
}) {
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [chatMessages, setChatMessages] = useState([INITIAL_MESSAGE]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    fetchOllamaModels().then((models) => {
      setAvailableModels(models);
      if (models.length > 0) setSelectedModel(models[0]);
    });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, chatLoading]);

  const handleSendChat = useCallback(async () => {
    const message = chatInput.trim();
    if (!message || chatLoading) return;

    setChatError(null);
    setChatMessages((prev) => [...prev, { role: "user", text: message }]);
    setChatInput("");
    setChatLoading(true);
    try {
      const reply = await sendChatMessage(message, selectedModel || undefined);
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: reply || "No response." },
      ]);
    } catch (e) {
      setChatError(e?.message || "Chat request failed");
    } finally {
      setChatLoading(false);
    }
  }, [chatInput, chatLoading, selectedModel]);

  return (
    <section className="bottom-panel-left" aria-label="Chat">
      <div>
        <button type="button" className="action-link" onClick={() => setLeftCollapsed(!leftCollapsed)}>
          {leftCollapsed ? "Show Chat" : "Hide Chat"}
        </button>
      </div>
      <Collapse isOpened={!leftCollapsed}>
        <div className="chatbot-section">
          {availableModels.length > 0 && (
            <div className="chat-model-row">
              <span className="chat-model-label">LLM</span>
              <select
                className="chat-model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {availableModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
          )}

          <div className="chat-window" aria-live="polite">
            {chatMessages.map((msg, idx) => (
              <div
                key={idx}
                className={[
                  "chat-message",
                  msg.role === "user" ? "chat-user" : "chat-assistant",
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
            <div ref={bottomRef} />
          </div>

          {chatError && <div className="chat-error">{chatError}</div>}

          <div className="chat-input-row">
            <input
              type="text"
              className="chat-input"
              placeholder="Type your message..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleSendChat(); }}
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
      </Collapse>
    </section>
  );
}

export default ChatPanel;
