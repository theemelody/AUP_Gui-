const API_BASE = import.meta.env.VITE_API_BASE || "/api";

function normalizeKey(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function readRowValue(row, aliases) {
  if (!row || typeof row !== "object") return "";
  const normalizedAliases = aliases.map((alias) => normalizeKey(alias));
  for (const [key, value] of Object.entries(row)) {
    if (normalizedAliases.includes(normalizeKey(key))) {
      return String(value ?? "").trim();
    }
  }
  return "";
}

function readRowRawValue(row, aliases) {
  if (!row || typeof row !== "object") return null;
  const normalizedAliases = aliases.map((alias) => normalizeKey(alias));
  for (const [key, value] of Object.entries(row)) {
    if (normalizedAliases.includes(normalizeKey(key))) {
      return value;
    }
  }
  return null;
}

function normalizeMapboxTypeList(value) {
  const rawValues = Array.isArray(value)
    ? value
    : String(value ?? "")
        .split(/[,;|]/)
        .map((item) => item.trim())
        .filter(Boolean);

  return Array.from(
    new Set(
      rawValues
        .map((item) => String(item || "").trim().toLowerCase())
        .filter(Boolean)
    )
  );
}

function normalizeConstructionMappingRows(payload) {
  const rawRows = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.rows)
      ? payload.rows
      : Array.isArray(payload?.data)
        ? payload.data
        : [];

  return rawRows
    .map((row) => {
      const yearStartRaw = readRowValue(row, ["year_start", "yearStart"]);
      const yearEndRaw = readRowValue(row, ["year_end", "yearEnd"]);
      const year_start = Number.parseInt(yearStartRaw, 10);
      const year_end = Number.parseInt(yearEndRaw, 10);

      return {
        const_type: readRowValue(row, ["const_type", "constType"]),
        year_start: Number.isFinite(year_start) ? year_start : null,
        year_end: Number.isFinite(year_end) ? year_end : null,
        refurbishment_type: readRowValue(row, [
          "refurbishment_type",
          "refurbishmentType",
          "refurbishment"
        ]),
        detail: readRowValue(row, ["detail", "detail_type", "detailType"]),
        mapbox_type: normalizeMapboxTypeList(
          readRowRawValue(row, ["mapbox_type", "mapboxType", "mapbox_types", "mapboxTypes"])
        ),
        cea_use_type1: readRowValue(row, ["cea_use_type1", "ceaUseType1", "use_type"])
          .toUpperCase()
      };
    })
    .filter(
      (row) =>
        row.const_type &&
        row.refurbishment_type &&
        row.detail &&
        row.cea_use_type1 &&
        Number.isFinite(row.year_start) &&
        Number.isFinite(row.year_end)
    );
}

export async function fetchBuildings() {
  // Loads full buildings GeoJSON from backend (used when not selecting directly from Mapbox).
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

export async function fetchMapboxCeaUseTypeMapping() {
  // Fetches mapbox_type -> cea_use_type1 mapping used by frontend selection pipeline.
  const res = await fetch(`${API_BASE}/mapbox-cea-use-type-mapping`);
  if (!res.ok) {
    let detail = `Mapping request failed (${res.status})`;
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

export async function fetchConstructionTypeMapping() {
  // Fetches decomposed CEA construction types used by right-panel feature definition.
  const res = await fetch(`${API_BASE}/construction-type-mapping`);
  if (!res.ok) {
    let detail = `Construction mapping request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data?.detail || detail;
    } catch {
      // Keep fallback detail message.
    }
    throw new Error(detail);
  }
  const data = await res.json();
  return normalizeConstructionMappingRows(data);
}

export async function selectBuildings(geometry, { signal } = {}) {
  // Generic selection endpoint; currently kept for backend-driven selection workflows.
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
  // Sends chat input to backend, which proxies to Ollama and returns model reply.
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

