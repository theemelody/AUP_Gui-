const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export async function fetchBuildings() {
  const res = await fetch(`${API_BASE}/buildings`);
  if (!res.ok) {
    let detail = `Buildings request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data?.detail || detail;
    } catch {
      // Keep fallback detail message.
    }
    throw new Error(detail);
  }
  const text = await res.text();
  return JSON.parse(text);
}

export async function selectBuildings(geometry, { signal } = {}) {
  const res = await fetch(`${API_BASE}/select`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ geometry }),
    signal
  });
  if (!res.ok) {
    let detail = `Selection request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data?.detail || detail;
    } catch {
      // Keep fallback detail message.
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function sendChatMessage(message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message })
  });
  if (!res.ok) {
    let detail = `Chat request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data?.detail || detail;
    } catch {
      // Keep fallback detail message.
    }
    throw new Error(detail);
  }
  const data = await res.json();
  return data.reply;
}

