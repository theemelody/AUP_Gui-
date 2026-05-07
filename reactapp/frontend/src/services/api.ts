import { normalizeMapboxTypeList } from '../utils/selection.js';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

type Row = Record<string, unknown>;

interface ConstructionMappingRow {
  const_type: string;
  year_start: number | null;
  year_end: number | null;
  refurbishment_type: string;
  detail: string;
  mapbox_type: string[];
  cea_use_type1: string;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function normalizeKey(value: unknown): string {
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
}

function readRowValue(row: Row, aliases: string[]): string {
  if (!row || typeof row !== 'object') return '';
  const normalizedAliases = aliases.map(normalizeKey);
  for (const [key, value] of Object.entries(row)) {
    if (normalizedAliases.includes(normalizeKey(key))) return String(value ?? '').trim();
  }
  return '';
}

function readRowRawValue(row: Row, aliases: string[]): unknown {
  if (!row || typeof row !== 'object') return null;
  const normalizedAliases = aliases.map(normalizeKey);
  for (const [key, value] of Object.entries(row)) {
    if (normalizedAliases.includes(normalizeKey(key))) return value;
  }
  return null;
}

function normalizeConstructionMappingRows(payload: unknown): ConstructionMappingRow[] {
  const rawRows: Row[] = Array.isArray(payload)
    ? (payload as Row[])
    : Array.isArray((payload as Record<string, unknown>)?.rows)
      ? ((payload as Record<string, unknown>).rows as Row[])
      : Array.isArray((payload as Record<string, unknown>)?.data)
        ? ((payload as Record<string, unknown>).data as Row[])
        : [];

  return rawRows
    .map((row) => {
      const yearStartRaw = readRowValue(row, ['year_start', 'yearStart']);
      const yearEndRaw = readRowValue(row, ['year_end', 'yearEnd']);
      const year_start = Number.parseInt(yearStartRaw, 10);
      const year_end = Number.parseInt(yearEndRaw, 10);

      return {
        const_type: readRowValue(row, ['const_type', 'constType']),
        year_start: Number.isFinite(year_start) ? year_start : null,
        year_end: Number.isFinite(year_end) ? year_end : null,
        refurbishment_type: readRowValue(row, ['refurbishment_type', 'refurbishmentType', 'refurbishment']),
        detail: readRowValue(row, ['detail', 'detail_type', 'detailType']),
        mapbox_type: normalizeMapboxTypeList(
          readRowRawValue(row, ['mapbox_type', 'mapboxType', 'mapbox_types', 'mapboxTypes'])
        ),
        cea_use_type1: readRowValue(row, ['cea_use_type1', 'ceaUseType1', 'use_type']).toUpperCase(),
      };
    })
    .filter(
      (row): row is ConstructionMappingRow =>
        Boolean(row.const_type) &&
        Boolean(row.refurbishment_type) &&
        Boolean(row.detail) &&
        Boolean(row.cea_use_type1) &&
        Number.isFinite(row.year_start) &&
        Number.isFinite(row.year_end)
    );
}

async function fetchJson<T>(url: string, init?: RequestInit, errorPrefix?: string): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${errorPrefix || 'Request'} failed (${res.status})`;
    try { const d = await res.json(); detail = (d as Record<string, string>)?.detail || detail; } catch { /* keep fallback */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ── API functions ─────────────────────────────────────────────────────────────

export async function fetchBuildings(): Promise<unknown> {
  // Loads full buildings GeoJSON from backend (used when not selecting directly from Mapbox).
  const res = await fetch(`${API_BASE}/buildings`);
  if (!res.ok) {
    let detail = `Buildings request failed (${res.status})`;
    try { const d = await res.json(); detail = (d as Record<string, string>)?.detail || detail; } catch { /* keep */ }
    throw new Error(detail);
  }
  return JSON.parse(await res.text());
}

export async function fetchMapboxCeaUseTypeMapping(): Promise<Record<string, string>> {
  return fetchJson<Record<string, string>>(`${API_BASE}/mapbox-cea-use-type-mapping`, undefined, 'Mapping request');
}

export async function fetchConstructionTypeMapping(): Promise<ConstructionMappingRow[]> {
  const data = await fetchJson<unknown>(`${API_BASE}/construction-type-mapping`, undefined, 'Construction mapping request');
  return normalizeConstructionMappingRows(data);
}

export async function selectBuildings(geometry: unknown, { signal }: { signal?: AbortSignal } = {}): Promise<unknown> {
  return fetchJson<unknown>(`${API_BASE}/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ geometry }),
    signal,
  }, 'Selection request');
}

export async function exportCeaShapefile(
  selectedGeoJSON: unknown,
  scenarioName = '',
  drawnPolygon: unknown = null,
): Promise<unknown> {
  return fetchJson<unknown>(`${API_BASE}/export-cea-shp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selected_geojson: selectedGeoJSON, scenario_name: scenarioName, site_polygon: drawnPolygon }),
  }, 'CEA export request');
}

export async function fetchScenarios(): Promise<string[]> {
  const data = await fetchJson<Record<string, unknown>>(`${API_BASE}/scenarios`, undefined, 'Scenarios request');
  return Array.isArray(data?.scenarios) ? (data.scenarios as string[]) : [];
}

export async function saveScenarioBuildings(
  selectedGeoJSON: unknown,
  scenarioName: string,
  drawnPolygon: unknown = null,
): Promise<{ success: boolean; scenario_path?: string }> {
  return fetchJson<{ success: boolean; scenario_path?: string }>(`${API_BASE}/save-scenario`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selected_geojson: selectedGeoJSON, scenario_name: scenarioName, site_polygon: drawnPolygon }),
  }, 'Save scenario request');
}

export function downloadBase64Zip(zipBase64: string, filename = 'cea_selected_buildings.zip'): void {
  if (!zipBase64) throw new Error('Missing ZIP payload');

  const binary = atob(zipBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const blob = new Blob([bytes], { type: 'application/zip' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export async function fetchOllamaModels(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/ollama-models`);
    if (!res.ok) return [];
    const data = await res.json() as Record<string, unknown>;
    return Array.isArray(data?.models) ? (data.models as string[]) : [];
  } catch {
    return [];
  }
}

export async function fetchScenarioStatus(scenarioName: string): Promise<string> {
  const res = await fetch(`${API_BASE}/scenario-status/${encodeURIComponent(scenarioName)}`);
  if (!res.ok) return 'missing';
  const data = await res.json() as Record<string, string>;
  return data?.status || 'missing';
}

export async function fetchScenarioData(scenarioName: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(
    `${API_BASE}/scenario-data/${encodeURIComponent(scenarioName)}`,
    undefined,
    'Scenario data not found'
  );
}

export async function fetchTechTreeGraph(scenarioName = '', region = 'DE'): Promise<unknown> {
  const params = new URLSearchParams({ region });
  if (scenarioName) params.append('scenario_name', scenarioName);
  return fetchJson<unknown>(`${API_BASE}/techtree-graph?${params}`, undefined, 'TechTree graph request');
}

export async function sendChatMessage(message: string, model?: string): Promise<string> {
  const data = await fetchJson<Record<string, string>>(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, ...(model ? { model } : {}) }),
  }, 'Chat request');
  return data.reply;
}

export interface KpiAnnualRow {
  name: string;
  GFA_m2: number;
  Aroof_m2?: number;
  Qhs_sys_MWhyr: number;
  Qww_sys_MWhyr: number;
  Qcs_sys_MWhyr: number;
  E_sys_MWhyr: number;
  QH_sys_MWhyr?: number;
  QC_sys_MWhyr?: number;
  E_sys0_kW?: number;
  Qhs_sys0_kW?: number;
  Qcs_sys0_kW?: number;
}

export interface KpiData {
  available: string[];
  meta: { building_count: number; total_gfa_m2: number };
  annual: KpiAnnualRow[] | null;
  monthly: { labels: string[]; heating_MWh: number[]; cooling_MWh: number[]; electricity_MWh: number[] } | null;
  monthly_balance: Record<string, number[] | string[]> | null;
  load_duration: { heating_kWh: number[]; cooling_kWh: number[]; electricity_kWh: number[] } | null;
  hourly_sample: Record<string, number[] | string[]> | null;
  solar_radiation: Record<string, number[] | string[]> | null;
  emissions: Record<string, unknown>[] | null;
  costs: Record<string, unknown>[] | null;
  potentials: Record<string, Record<string, unknown>[]> | null;
  network: Record<string, unknown> | null;
}

export async function fetchKpiData(scenarioName: string): Promise<KpiData> {
  return fetchJson<KpiData>(
    `${API_BASE}/kpi-data/${encodeURIComponent(scenarioName)}`,
    undefined,
    'KPI data request',
  );
}
