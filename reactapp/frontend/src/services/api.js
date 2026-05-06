import { normalizeMapboxTypeList } from "../utils/selection.js";

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

export async function exportCeaShapefile(selectedGeoJSON, scenarioName = "", drawnPolygon = null) {
  // Converts selected Mapbox GeoJSON into a projected, CEA-compatible shapefile ZIP.
  // drawnPolygon is the area selection boundary to use for site.shp
  const res = await fetch(`${API_BASE}/export-cea-shp`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      selected_geojson: selectedGeoJSON,
      scenario_name: scenarioName,
      site_polygon: drawnPolygon
    })
  });
  if (!res.ok) {
    let detail = `CEA export request failed (${res.status})`;
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

export async function fetchScenarios() {
  const res = await fetch(`${API_BASE}/scenarios`);
  if (!res.ok) {
    let detail = `Scenarios request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data?.detail || detail;
    } catch {
      // Keep fallback detail message.
    }
    throw new Error(detail);
  }
  const data = await res.json();
  return Array.isArray(data?.scenarios) ? data.scenarios : [];
}

export async function saveScenarioBuildings(selectedGeoJSON, scenarioName, drawnPolygon = null) {
  // Saves building selection as uncompressed files to scenarios folder.
  // scenarioName is required - will create scenario-name-scenario folder structure
  const res = await fetch(`${API_BASE}/save-scenario`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      selected_geojson: selectedGeoJSON,
      scenario_name: scenarioName,
      site_polygon: drawnPolygon
    })
  });
  if (!res.ok) {
    let detail = `Save scenario request failed (${res.status})`;
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

export function downloadBase64Zip(zipBase64, filename = "cea_selected_buildings.zip") {
  if (!zipBase64) {
    throw new Error("Missing ZIP payload")
  }

  const binary = atob(zipBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  const blob = new Blob([bytes], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export async function fetchOllamaModels() {
  try {
    const res = await fetch(`${API_BASE}/ollama-models`);
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data?.models) ? data.models : [];
  } catch {
    return [];
  }
}

export async function fetchScenarioStatus(scenarioName) {
  const res = await fetch(`${API_BASE}/scenario-status/${encodeURIComponent(scenarioName)}`);
  if (!res.ok) return "missing";
  const data = await res.json();
  return data?.status || "missing";
}

export async function fetchScenarioData(scenarioName) {
  const res = await fetch(`${API_BASE}/scenario-data/${encodeURIComponent(scenarioName)}`);
  if (!res.ok) throw new Error(`Scenario data not found (${res.status})`);
  return res.json();
}

export async function fetchTechTreeGraph(scenarioName = '', region = 'DE') {
  const params = new URLSearchParams({ region });
  if (scenarioName) params.append('scenario_name', scenarioName);
  const res = await fetch(`${API_BASE}/techtree-graph?${params}`);
  if (!res.ok) {
    let detail = `TechTree graph request failed (${res.status})`;
    try { const d = await res.json(); detail = d?.detail || detail; } catch { /* keep fallback */ }
    throw new Error(detail);
  }
  return res.json();
}

export async function sendChatMessage(message, model) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message, ...(model ? { model } : {}) })
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

