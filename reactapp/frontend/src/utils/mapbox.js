import booleanIntersects from "@turf/boolean-intersects";
import { getFeatureStableKey } from "./selection.js";

// ── building type helpers ─────────────────────────────────────────────────────

export function getMapboxBuildingType(properties) {
  const candidates = [
    properties?.class,
    properties?.building,
    properties?.subclass,
    properties?.type,
  ];

  for (const candidate of candidates) {
    const value = String(candidate || "").trim().toLowerCase();
    if (!value) continue;
    if (value === "building") continue;
    return value;
  }

  const fallback = String(
    properties?.type || properties?.class || properties?.building || ""
  )
    .trim()
    .toLowerCase();
  return fallback || null;
}

export function getBuildingFeatureKey(feature) {
  const properties = feature?.properties || {};
  const buildingId = properties?.building_id ?? properties?.buildingId;
  if (buildingId !== null && buildingId !== undefined && buildingId !== "") {
    return String(buildingId);
  }
  if (feature?.id !== null && feature?.id !== undefined && feature?.id !== "") {
    return String(feature.id);
  }
  return null;
}

export function buildBuildingTypeLookup(features) {
  const lookup = new Map();
  for (const feature of features) {
    const properties = feature?.properties || {};
    const mapboxType = getMapboxBuildingType(properties);
    const featureKey = getBuildingFeatureKey(feature);
    if (!featureKey) continue;
    if (mapboxType && mapboxType !== "building" && mapboxType !== "building:part") {
      lookup.set(featureKey, mapboxType);
    }
  }
  return lookup;
}

export function mapToCeaUseType1(mapboxType, mapping) {
  if (!mapboxType) return null;
  return mapping?.[mapboxType] || null;
}

export function estimateFloors(properties) {
  // Use explicit floor count when present; fallback to height / 3m approximation.
  const explicitLevels = Number(properties?.levels ?? properties?.num_floors);
  if (Number.isFinite(explicitLevels) && explicitLevels > 0) {
    return Math.round(explicitLevels);
  }
  const height = Number(properties?.height);
  const minHeight = Number(properties?.min_height ?? 0);
  if (Number.isFinite(height) && height > minHeight) {
    return Math.max(1, Math.round((height - minHeight) / 3));
  }
  return null;
}

export function inferCeaUseType1(properties, mapboxType) {
  const floors = estimateFloors(properties);
  const height = Number(properties?.height);
  const genericBuilding = !mapboxType || mapboxType === "building";

  if (genericBuilding) {
    if (Number.isFinite(floors)) return floors <= 2 ? "SINGLE_RES" : "MULTI_RES";
    if (Number.isFinite(height)) return height < 10 ? "SINGLE_RES" : "MULTI_RES";
    return "MULTI_RES";
  }

  if (Number.isFinite(floors)) {
    if (floors <= 2) return "SINGLE_RES";
    if (floors <= 5) return "MULTI_RES";
  }
  if (Number.isFinite(height)) {
    if (height < 10) return "SINGLE_RES";
    if (height < 20) return "MULTI_RES";
  }
  return null;
}

// ── geometry helpers ──────────────────────────────────────────────────────────

export function normalizeDrawFeatures(geometry) {
  // Ensure downstream logic always receives a list of GeoJSON Features.
  if (!geometry || typeof geometry !== "object") return [];
  if (geometry.type === "FeatureCollection") {
    return (geometry.features || []).filter(
      (f) => f?.type === "Feature" && f?.geometry
    );
  }
  if (geometry.type === "Feature" && geometry.geometry) return [geometry];
  return [{ type: "Feature", properties: {}, geometry }];
}

export function dedupeFeatures(features) {
  // querySourceFeatures can return duplicates across tiles, so dedupe by id/geometry.
  const seen = new Set();
  const out = [];
  for (const feature of features) {
    const key =
      feature?.id != null
        ? String(feature.id)
        : JSON.stringify(feature?.geometry || {});
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(feature);
  }
  return out;
}

export function toPlainGeometry(geometry) {
  if (!geometry) return null;
  const { type, coordinates } = geometry;
  if (!type || !coordinates) return null;
  return { type, coordinates: JSON.parse(JSON.stringify(coordinates)) };
}

export function getBoundsFromGeoJSON(geojson) {
  if (!geojson) return null;
  let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;

  const visit = (coords) => {
    if (Array.isArray(coords[0]) && typeof coords[0][0] === "number") {
      coords.forEach((c) => visit(c));
      return;
    }
    if (Array.isArray(coords[0]) && Array.isArray(coords[0][0])) {
      coords.forEach((ring) => visit(ring));
      return;
    }
    const [lng, lat] = coords;
    if (typeof lng === "number" && typeof lat === "number") {
      minLng = Math.min(minLng, lng);
      minLat = Math.min(minLat, lat);
      maxLng = Math.max(maxLng, lng);
      maxLat = Math.max(maxLat, lat);
    }
  };

  const walk = (obj) => {
    if (!obj) return;
    if (obj.type === "FeatureCollection" && obj.features) { obj.features.forEach((f) => walk(f)); return; }
    if (obj.type === "Feature" && obj.geometry) { walk(obj.geometry); return; }
    if (obj.type === "GeometryCollection" && obj.geometries) { obj.geometries.forEach((g) => walk(g)); return; }
    if (obj.type === "Polygon" && obj.coordinates) visit(obj.coordinates);
    if (obj.type === "MultiPolygon" && obj.coordinates) obj.coordinates.forEach((p) => visit(p));
  };

  walk(geojson);
  if (minLng === Infinity) return null;
  return [[minLng, minLat], [maxLng, maxLat]];
}

// ── feature normalization ─────────────────────────────────────────────────────

export function normalizeMapboxFeature(feature, mapping, index = 0) {
  const properties = feature?.properties || {};
  const height = Number(properties.height);
  const minHeight = Number(properties.min_height ?? 0);
  const rawMapboxType = getMapboxBuildingType(properties);
  const resolvedMapboxType =
    rawMapboxType && rawMapboxType !== "building" && rawMapboxType !== "building:part"
      ? rawMapboxType
      : null;
  const mappedUseType =
    mapToCeaUseType1(resolvedMapboxType || rawMapboxType, mapping) ||
    String(properties?.cea_use_type1 || "").trim().toUpperCase() ||
    inferCeaUseType1(properties, resolvedMapboxType || rawMapboxType) ||
    "UNKNOWN";
  const stableKey = getFeatureStableKey(feature, index);
  const geometry = toPlainGeometry(feature?.geometry);
  if (!geometry) return null;

  return {
    type: "Feature",
    id: feature?.id ?? stableKey,
    geometry,
    properties: {
      ...properties,
      __selection_key: stableKey,
      mapbox_type: resolvedMapboxType || rawMapboxType,
      mapbox_type_raw: rawMapboxType,
      cea_use_type1: mappedUseType,
      height: Number.isFinite(height) ? height : null,
      min_height: Number.isFinite(minHeight) ? minHeight : 0,
      estimated_floors: estimateFloors(properties),
    },
  };
}

// ── selection ─────────────────────────────────────────────────────────────────

export function selectBuildingsFromMapbox(map, geometry, mapping) {
  const drawFeatures = normalizeDrawFeatures(geometry);
  if (!drawFeatures.length) {
    return { count: 0, selected_geojson: null, zip_base64: null, buildings: [] };
  }

  const candidates = dedupeFeatures(
    map.querySourceFeatures("composite", { sourceLayer: "building" })
  );

  const selectedFeatures = candidates.filter((candidate) => {
    if (!candidate?.geometry) return false;
    return drawFeatures.some((drawFeature) => {
      try { return booleanIntersects(candidate, drawFeature); }
      catch { return false; }
    });
  });

  const typeLookup = buildBuildingTypeLookup(selectedFeatures);
  const normalizedFeatures = selectedFeatures
    .map((feature, index) => {
      const normalizedFeature = normalizeMapboxFeature(feature, mapping, index);
      if (!normalizedFeature) return null;

      const featureKey = getBuildingFeatureKey(feature);
      const resolvedType =
        (featureKey && typeLookup.get(featureKey)) ||
        normalizedFeature.properties?.mapbox_type ||
        null;

      if (
        resolvedType &&
        resolvedType !== normalizedFeature.properties.mapbox_type &&
        (normalizedFeature.properties.mapbox_type === "building:part" ||
          normalizedFeature.properties.mapbox_type === "building" ||
          !normalizedFeature.properties.mapbox_type)
      ) {
        normalizedFeature.properties.mapbox_type = resolvedType;
        normalizedFeature.properties.cea_use_type1 =
          mapToCeaUseType1(resolvedType, mapping) ||
          String(normalizedFeature.properties?.cea_use_type1 || "").trim().toUpperCase() ||
          inferCeaUseType1(normalizedFeature.properties, resolvedType) ||
          normalizedFeature.properties.cea_use_type1;
      }

      return normalizedFeature;
    })
    .filter(Boolean);

  return {
    count: normalizedFeatures.length,
    buildings: normalizedFeatures.map((f) => f.properties || {}),
    selected_geojson:
      normalizedFeatures.length > 0
        ? { type: "FeatureCollection", features: normalizedFeatures }
        : null,
    zip_base64: null,
  };
}

export function selectBuildingsFromFeatureCollection(baseFeatureCollection, geometry) {
  const drawFeatures = normalizeDrawFeatures(geometry);
  const baseFeatures = Array.isArray(baseFeatureCollection?.features)
    ? baseFeatureCollection.features.filter((f) => f?.geometry)
    : [];

  if (!drawFeatures.length || !baseFeatures.length) {
    return { count: 0, selected_geojson: null, buildings: [], building_keys: [] };
  }

  const selectedFeatures = baseFeatures.filter((candidate) =>
    drawFeatures.some((drawFeature) => {
      try { return booleanIntersects(candidate, drawFeature); }
      catch { return false; }
    })
  );

  const buildingKeys = selectedFeatures.map((feature, index) =>
    getFeatureStableKey(feature, index)
  );

  return {
    count: selectedFeatures.length,
    selected_geojson:
      selectedFeatures.length > 0
        ? { type: "FeatureCollection", features: selectedFeatures }
        : null,
    buildings: selectedFeatures.map((feature) => ({ ...(feature?.properties || {}) })),
    building_keys: buildingKeys,
  };
}

// ── layer definitions ─────────────────────────────────────────────────────────

export const buildingsFillLayer = {
  id: "buildings-fill",
  type: "fill",
  paint: { "fill-color": "#3186cc", "fill-opacity": 0.35 },
};

export const buildingsOutlineLayer = {
  id: "buildings-outline",
  type: "line",
  paint: { "line-color": "#0f172a", "line-width": 1 },
};

export const selectedFillLayer = {
  id: "selected-fill",
  type: "fill",
  paint: { "fill-color": "#f97316", "fill-opacity": 0.45 },
};

export const selectedOutlineLayer = {
  id: "selected-outline",
  type: "line",
  paint: { "line-color": "#7c2d12", "line-width": 2 },
};

export const lockedBuildingsLayer = {
  id: "locked-extrusion",
  type: "fill-extrusion",
  paint: {
    "fill-extrusion-color": [
      "match",
      ["coalesce", ["get", "__assignment_state"], "pending"],
      "defined", "#eab308",
      "complete", "#22c55e",
      "#f97316",
    ],
    "fill-extrusion-opacity": 0.85,
    "fill-extrusion-height": ["coalesce", ["to-number", ["get", "height"]], 12],
    "fill-extrusion-base": ["coalesce", ["to-number", ["get", "min_height"]], 0],
  },
};

export const lockedFillLayer = {
  id: "locked-fill",
  type: "fill",
  paint: {
    "fill-color": [
      "match",
      ["coalesce", ["get", "__assignment_state"], "pending"],
      "defined", "#eab308",
      "complete", "#22c55e",
      "#f97316",
    ],
    "fill-opacity": 0.6,
  },
};

export const lockedOutlineLayer = {
  id: "locked-outline",
  type: "line",
  paint: {
    "line-color": [
      "match",
      ["coalesce", ["get", "__assignment_state"], "pending"],
      "defined", "#92400e",
      "complete", "#166534",
      "#7c2d12",
    ],
    "line-width": 2,
  },
};
