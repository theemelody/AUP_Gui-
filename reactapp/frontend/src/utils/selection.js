export function parseGeoJSON(value) {
  if (!value) return null;
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

export function normalizeMapboxType(value) {
  return String(value || "").trim().toLowerCase();
}

export function normalizeMapboxTypeList(value) {
  const rawValues = Array.isArray(value)
    ? value
    : String(value ?? "")
        .split(/[,;|]/)
        .map((item) => item.trim())
        .filter(Boolean);

  return Array.from(
    new Set(rawValues.map((item) => normalizeMapboxType(item)).filter(Boolean))
  );
}

export function getBuildingMapboxType(building) {
  return normalizeMapboxType(
    building?.mapbox_type || building?.mapboxType || building?.type || building?.class || building?.building
  );
}

export function getFeatureStableKey(feature, index = 0) {
  const props = feature?.properties || {};
  const explicitKey = props.__selection_key;
  if (explicitKey) return String(explicitKey);

  const candidates = [
    feature?.id,
    props.id,
    props.osm_id,
    props.osm_way_id,
    props.mapbox_id
  ];

  for (const value of candidates) {
    if (value !== null && value !== undefined && value !== "") {
      return String(value);
    }
  }

  return `geom-${index}-${JSON.stringify(feature?.geometry || {})}`;
}
