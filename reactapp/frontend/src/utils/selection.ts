export function parseGeoJSON(value: unknown): unknown {
  if (!value) return null;
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

export function normalizeMapboxType(value: unknown): string {
  return String(value || '').trim().toLowerCase();
}

export function normalizeMapboxTypeList(value: unknown): string[] {
  const rawValues = Array.isArray(value)
    ? (value as unknown[])
    : String(value ?? '')
        .split(/[,;|]/)
        .map((item) => item.trim())
        .filter(Boolean);

  return Array.from(
    new Set(rawValues.map((item) => normalizeMapboxType(item)).filter(Boolean))
  );
}

export function getBuildingMapboxType(building: Record<string, unknown>): string {
  return normalizeMapboxType(
    building?.mapbox_type ||
      building?.mapboxType ||
      building?.type ||
      building?.class ||
      building?.building
  );
}

export function getFeatureStableKey(feature: unknown, index = 0): string {
  const f = feature as { id?: unknown; properties?: Record<string, unknown>; geometry?: unknown };
  const props = f?.properties || {};
  const explicitKey = props.__selection_key;
  if (explicitKey) return String(explicitKey);

  const candidates = [f?.id, props.id, props.osm_id, props.osm_way_id, props.mapbox_id];
  for (const value of candidates) {
    if (value !== null && value !== undefined && value !== '') return String(value);
  }

  return `geom-${index}-${JSON.stringify(f?.geometry || {})}`;
}
