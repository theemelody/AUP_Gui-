import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import Map, {
  FullscreenControl,
  NavigationControl,
  Source,
  Layer
} from "react-map-gl";
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import booleanIntersects from "@turf/boolean-intersects";
import { fetchMapboxCeaUseTypeMapping } from "../services/api.js";

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_ACCESS_TOKEN;

function getMapboxBuildingType(properties) {
  // Mapbox building features can expose category in different fields depending on source.
  const raw = properties?.type || properties?.class || properties?.building || null;
  if (!raw) return null;
  return String(raw).trim().toLowerCase();
}

function mapToCeaUseType1(mapboxType, mapping) {
  if (!mapboxType) return null;
  return mapping?.[mapboxType] || null;
}

function normalizeGeoJSON(value) {
  // Accept either already-parsed GeoJSON or serialized JSON strings.
  if (!value) return null;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }
  return value;
}

function estimateFloors(properties) {
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

function normalizeDrawFeatures(geometry) {
  // Ensure downstream logic always receives a list of GeoJSON Features.
  if (!geometry || typeof geometry !== "object") return [];

  if (geometry.type === "FeatureCollection") {
    return (geometry.features || []).filter(
      (f) => f?.type === "Feature" && f?.geometry
    );
  }

  if (geometry.type === "Feature" && geometry.geometry) {
    return [geometry];
  }

  return [{ type: "Feature", properties: {}, geometry }];
}

function dedupeFeatures(features) {
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

function toPlainGeometry(geometry) {
  if (!geometry) return null;
  const type = geometry?.type;
  const coordinates = geometry?.coordinates;
  if (!type || !coordinates) return null;
  return {
    type,
    coordinates: JSON.parse(JSON.stringify(coordinates))
  };
}

function normalizeMapboxFeature(feature, mapping, index = 0) {
  const properties = feature?.properties || {};
  const height = Number(properties.height);
  const minHeight = Number(properties.min_height ?? 0);
  const mapboxType = getMapboxBuildingType(properties);
  const mappedUseType = mapToCeaUseType1(mapboxType, mapping) || "MULTI_RES";
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
      cea_use_type1: mappedUseType,
      height: Number.isFinite(height) ? height : null,
      min_height: Number.isFinite(minHeight) ? minHeight : 0,
      estimated_floors: estimateFloors(properties)
    }
  };
}

function getFeatureStableKey(feature, index = 0) {
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

function selectBuildingsFromMapbox(map, geometry, mapping) {
  // Build a selected subset directly from the basemap building layer.
  const drawFeatures = normalizeDrawFeatures(geometry);
  if (!drawFeatures.length) {
    return {
      count: 0,
      selected_geojson: null,
      zip_base64: null,
      buildings: []
    };
  }

  const candidates = dedupeFeatures(
    map.querySourceFeatures("composite", { sourceLayer: "building" })
  );

  const selectedFeatures = candidates.filter((candidate) => {
    if (!candidate?.geometry) return false;
    return drawFeatures.some((drawFeature) => {
      try {
        return booleanIntersects(candidate, drawFeature);
      } catch {
        return false;
      }
    });
  });

  const normalizedFeatures = selectedFeatures
    .map((feature, index) => normalizeMapboxFeature(feature, mapping, index))
    .filter(Boolean);

  const buildings = normalizedFeatures.map((feature) => feature.properties || {});

  return {
    count: normalizedFeatures.length,
    buildings,
    selected_geojson:
      normalizedFeatures.length > 0
        ? { type: "FeatureCollection", features: normalizedFeatures }
        : null,
    zip_base64: null
  };
}

function selectBuildingsFromFeatureCollection(baseFeatureCollection, geometry) {
  // Select only from already confirmed buildings when in construction phase.
  const drawFeatures = normalizeDrawFeatures(geometry);
  const baseFeatures = Array.isArray(baseFeatureCollection?.features)
    ? baseFeatureCollection.features.filter((f) => f?.geometry)
    : [];

  if (!drawFeatures.length || !baseFeatures.length) {
    return {
      count: 0,
      selected_geojson: null,
      buildings: [],
      building_keys: []
    };
  }

  const selectedFeatures = baseFeatures.filter((candidate) => {
    return drawFeatures.some((drawFeature) => {
      try {
        return booleanIntersects(candidate, drawFeature);
      } catch {
        return false;
      }
    });
  });

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
    building_keys: buildingKeys
  };
}

/** Returns bounds [[minLng, minLat], [maxLng, maxLat]] from GeoJSON. */
function getBoundsFromGeoJSON(geojson) {
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
    if (obj.type === "FeatureCollection" && obj.features) {
      obj.features.forEach((f) => walk(f));
      return;
    }
    if (obj.type === "Feature" && obj.geometry) walk(obj.geometry);
    if (obj.type === "GeometryCollection" && obj.geometries) {
      obj.geometries.forEach((g) => walk(g));
      return;
    }
    if (obj.type === "Polygon" && obj.coordinates) visit(obj.coordinates);
    if (obj.type === "MultiPolygon" && obj.coordinates) {
      obj.coordinates.forEach((p) => visit(p));
    }
  };
  walk(geojson);
  if (minLng === Infinity) return null;
  return [
    [minLng, minLat],
    [maxLng, maxLat]
  ];
}

const buildingsFillLayer = {
  id: "buildings-fill",
  type: "fill",
  paint: {
    "fill-color": "#3186cc",
    "fill-opacity": 0.35
  }
};

const buildingsOutlineLayer = {
  id: "buildings-outline",
  type: "line",
  paint: {
    "line-color": "#0f172a",
    "line-width": 1
  }
};

const selectedFillLayer = {
  id: "selected-fill",
  type: "fill",
  paint: {
    "fill-color": "#f97316",
    "fill-opacity": 0.45
  }
};

const selectedOutlineLayer = {
  id: "selected-outline",
  type: "line",
  paint: {
    "line-color": "#7c2d12",
    "line-width": 2
  }
};

const lockedBuildingsLayer = {
  id: "locked-extrusion",
  type: "fill-extrusion",
  paint: {
    "fill-extrusion-color": [
      "match",
      ["coalesce", ["get", "__assignment_state"], "pending"],
      "defined",
      "#eab308",
      "complete",
      "#22c55e",
      "#f97316"
    ],
    "fill-extrusion-opacity": 0.85,
    "fill-extrusion-height": ["coalesce", ["to-number", ["get", "height"]], 12],
    "fill-extrusion-base": ["coalesce", ["to-number", ["get", "min_height"]], 0]
  }
};

const lockedFillLayer = {
  id: "locked-fill",
  type: "fill",
  paint: {
    "fill-color": [
      "match",
      ["coalesce", ["get", "__assignment_state"], "pending"],
      "defined",
      "#eab308",
      "complete",
      "#22c55e",
      "#f97316"
    ],
    "fill-opacity": 0.6
  }
};

const lockedOutlineLayer = {
  id: "locked-outline",
  type: "line",
  paint: {
    "line-color": [
      "match",
      ["coalesce", ["get", "__assignment_state"], "pending"],
      "defined",
      "#92400e",
      "complete",
      "#166534",
      "#7c2d12"
    ],
    "line-width": 2
  }
};

function MapView({
  buildingsGeoJSON,
  selectedGeoJSON,
  lockedSelectionGeoJSON,
  selectionLocked,
  constructionPhaseActive,
  onSelection,
  onConstructionAreaSelection
}) {
  const mapRef = useRef(null);
  const [viewState, setViewState] = useState({
    longitude: 2.1734,
    latitude: 41.3851,
    zoom: 12.5,
    pitch: 60,
    bearing: -20
  });
  const [mapLoaded, setMapLoaded] = useState(false);

  const geojson = useMemo(
    () => normalizeGeoJSON(buildingsGeoJSON),
    [buildingsGeoJSON]
  );
  const normalizedSelectedGeoJSON = useMemo(
    () => normalizeGeoJSON(selectedGeoJSON),
    [selectedGeoJSON]
  );
  const normalizedLockedSelectionGeoJSON = useMemo(
    () => normalizeGeoJSON(lockedSelectionGeoJSON),
    [lockedSelectionGeoJSON]
  );
  const hasFittedRef = useRef(false);
  const drawRef = useRef(null);
  const selectDebounceRef = useRef(null);
  const mapboxCeaUseTypeMappingRef = useRef({});
  const mapboxMappingLoadedRef = useRef(false);
  const selectionLockedRef = useRef(selectionLocked);
  const constructionPhaseActiveRef = useRef(constructionPhaseActive);
  const lockedSelectionGeoJSONRef = useRef(normalizedLockedSelectionGeoJSON);
  const onSelectionRef = useRef(onSelection);
  const onConstructionAreaSelectionRef = useRef(onConstructionAreaSelection);
  const savedPitchRef = useRef(null);

  useEffect(() => {
    selectionLockedRef.current = selectionLocked;
  }, [selectionLocked]);

  useEffect(() => {
    constructionPhaseActiveRef.current = constructionPhaseActive;
  }, [constructionPhaseActive]);

  useEffect(() => {
    lockedSelectionGeoJSONRef.current = normalizedLockedSelectionGeoJSON;
  }, [normalizedLockedSelectionGeoJSON]);

  useEffect(() => {
    onSelectionRef.current = onSelection;
  }, [onSelection]);

  useEffect(() => {
    onConstructionAreaSelectionRef.current = onConstructionAreaSelection;
  }, [onConstructionAreaSelection]);

  useEffect(() => {
    // Load mapbox->CEA mapping once; selection falls back to MULTI_RES if unavailable.
    if (mapboxMappingLoadedRef.current) return;
    mapboxMappingLoadedRef.current = true;
    let cancelled = false;
    fetchMapboxCeaUseTypeMapping()
      .then((mapping) => {
        if (!cancelled && mapping && typeof mapping === "object") {
          mapboxCeaUseTypeMappingRef.current = mapping;
        }
      })
      .catch(() => {
        // Keep fallback behavior when mapping CSV is unavailable.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const fitMapToBounds = useCallback(() => {
    // Fit map to static buildings extent on first load in SHP mode.
    if (!geojson || !mapRef.current) return;
    const bounds = getBoundsFromGeoJSON(geojson);
    if (!bounds) return;
    const map = mapRef.current.getMap();
    if (map) {
      map.fitBounds(bounds, { padding: 60, maxZoom: 18 });
      hasFittedRef.current = true;
    }
  }, [geojson]);

  useEffect(() => {
    if (!geojson || hasFittedRef.current) return;
    fitMapToBounds();
  }, [geojson, fitMapToBounds]);

  const has3DInitRef = useRef(false);

  const init3D = useCallback(() => {
    const map = mapRef.current?.getMap?.();
    if (!map || has3DInitRef.current) return;

    // Terrain (DEM) + sky
    if (!map.getSource("mapbox-dem")) {
      map.addSource("mapbox-dem", {
        type: "raster-dem",
        url: "mapbox://mapbox.mapbox-terrain-dem-v1",
        tileSize: 512,
        maxzoom: 14
      });
    }
    if (!map.getTerrain()) {
      map.setTerrain({ source: "mapbox-dem", exaggeration: 1.2 });
    }
    if (!map.getLayer("sky")) {
      map.addLayer({
        id: "sky",
        type: "sky",
        paint: {
          "sky-type": "atmosphere",
          "sky-atmosphere-sun": [0.0, 0.0],
          "sky-atmosphere-sun-intensity": 12
        }
      });
    }

    // 3D Buildings (Mapbox basemap "composite" source)
    if (!map.getLayer("3d-buildings") && map.getSource("composite")) {
      const labelLayerId = map
        .getStyle()
        ?.layers?.find((l) => l.type === "symbol" && l.layout?.["text-field"])
        ?.id;

      map.addLayer(
        {
          id: "3d-buildings",
          source: "composite",
          "source-layer": "building",
          type: "fill-extrusion",
          minzoom: 15,
          filter: ["==", ["get", "extrude"], "true"],
          paint: {
            "fill-extrusion-color": "#94a3b8",
            "fill-extrusion-opacity": 0.6,
            "fill-extrusion-height": ["get", "height"],
            "fill-extrusion-base": ["get", "min_height"]
          }
        },
        labelLayerId
      );

      map.setLayoutProperty(
        "3d-buildings",
        "visibility",
        selectionLocked ? "none" : "visible"
      );
    }

    has3DInitRef.current = true;
  }, [selectionLocked]);

  const ensureDrawControl = useCallback(() => {
    const map = mapRef.current?.getMap?.();
    if (!map || drawRef.current) return;

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: {
        polygon: true,
        trash: true
      }
    });
    map.addControl(draw, "top-left");
    drawRef.current = draw;

    const fireSelection = (geometry) => {
      // Drawing is used for initial selection, then reused for construction-area selection.
      const isLocked = selectionLockedRef.current;
      const isConstructionPhase = constructionPhaseActiveRef.current;
      if (isLocked && !isConstructionPhase) return;
      if (selectDebounceRef.current) clearTimeout(selectDebounceRef.current);
      selectDebounceRef.current = setTimeout(async () => {
        try {
          if (!geometry) {
            if (isLocked && isConstructionPhase) {
              onConstructionAreaSelectionRef.current?.({
                count: 0,
                selected_geojson: null,
                buildings: [],
                building_keys: [],
                selection_error: null
              });
            } else {
              onSelectionRef.current?.({
                count: 0,
                selected_geojson: null,
                zip_base64: null,
                buildings: [],
                selection_error: null
              });
            }
            return;
          }

          if (isLocked && isConstructionPhase) {
            const result = selectBuildingsFromFeatureCollection(
              lockedSelectionGeoJSONRef.current,
              geometry
            );
            onConstructionAreaSelectionRef.current?.({
              ...result,
              selection_error: null
            });
          } else {
            const result = selectBuildingsFromMapbox(
              map,
              geometry,
              mapboxCeaUseTypeMappingRef.current
            );
            onSelectionRef.current?.({ ...result, selection_error: null });
          }
        } catch (e) {
          if (e?.name === "AbortError") return;
          if (isLocked && isConstructionPhase) {
            onConstructionAreaSelectionRef.current?.({
              count: 0,
              selected_geojson: null,
              buildings: [],
              building_keys: [],
              selection_error: e?.message || "Construction area selection failed"
            });
          } else {
            onSelectionRef.current?.({
              count: 0,
              selected_geojson: null,
              zip_base64: null,
              buildings: [],
              selection_error: e?.message || "Selection failed"
            });
          }
        }
      }, 250);
    };

    const handleChange = () => {
      if (selectionLockedRef.current && !constructionPhaseActiveRef.current) return;
      const fc = draw.getAll();
      if (!fc?.features?.length) {
        fireSelection(null);
        return;
      }
      fireSelection(fc);
    };

    map.on("draw.create", handleChange);
    map.on("draw.update", handleChange);
    map.on("draw.delete", () => fireSelection(null));
  }, []);

  const handleMapLoad = useCallback(() => {
    init3D();
    ensureDrawControl();
    setMapLoaded(true);
    if (geojson && !hasFittedRef.current) {
      fitMapToBounds();
    }
  }, [ensureDrawControl, geojson, fitMapToBounds, init3D]);

  useEffect(() => {
    // Toggle visibility of raw 3D buildings once a selection is confirmed.
    const map = mapRef.current?.getMap?.();
    if (!map) return;
    const layer = map.getLayer("3d-buildings");
    if (layer) {
      map.setLayoutProperty("3d-buildings", "visibility", selectionLocked ? "none" : "visible");
    }
  }, [selectionLocked]);

  useEffect(() => {
    // Disable drawing only after lock if construction phase is not active.
    const map = mapRef.current?.getMap?.();
    const draw = drawRef.current;
    if (!map || !draw) return;

    const showDraw = !selectionLocked || constructionPhaseActive;
    const displayValue = showDraw ? "block" : "none";

    [".mapbox-gl-draw_ctrl-draw-btn", ".mapbox-gl-draw_ctrl-draw-btn.mapbox-gl-draw_polygon", ".mapbox-gl-draw_trash"].forEach(
      (selector) => {
        map.getContainer().querySelectorAll(selector).forEach((el) => {
          el.style.display = displayValue;
        });
      }
    );
  }, [constructionPhaseActive, selectionLocked]);

  useEffect(() => {
    if (!mapLoaded) return;
    const map = mapRef.current?.getMap?.();
    if (!map) return;

    // In construction phase, flatten pitch so draw vertices align with pointer on the ground plane.
    if (constructionPhaseActive) {
      if (savedPitchRef.current === null) {
        savedPitchRef.current = map.getPitch();
      }
      map.easeTo({ pitch: 0, duration: 180 });
      return;
    }

    if (savedPitchRef.current !== null) {
      map.easeTo({ pitch: savedPitchRef.current, duration: 180 });
      savedPitchRef.current = null;
    }
  }, [constructionPhaseActive, mapLoaded]);

  useEffect(() => {
    // Reset draw shapes when switching phase so stale polygons do not leak between modes.
    const draw = drawRef.current;
    if (!draw) return;
    draw.deleteAll();
  }, [selectionLocked]);

  useEffect(() => {
    ensureDrawControl();
    return () => {
      if (selectDebounceRef.current) clearTimeout(selectDebounceRef.current);
    };
  }, [ensureDrawControl]);

  useEffect(() => {
    if (!mapLoaded) return;
    const map = mapRef.current?.getMap?.();
    if (!map) return;

    const container = map.getContainer();
    if (!container) return;

    map.resize();

    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      map.resize();
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
    };
  }, [mapLoaded]);

  return (
    <div className="map-container">
      <Map
        ref={mapRef}
        {...viewState}
        onLoad={handleMapLoad}
        onMove={(evt) => setViewState(evt.viewState)}
        mapStyle="mapbox://styles/mapbox/light-v11"
        mapboxAccessToken={MAPBOX_TOKEN}
      >
        <NavigationControl position="top-left" />
        <FullscreenControl position="top-left" />

        {geojson && (
          <Source id="buildings" type="geojson" data={geojson}>
            <Layer {...buildingsFillLayer} />
            <Layer {...buildingsOutlineLayer} />
          </Source>
        )}

        {!selectionLocked && normalizedSelectedGeoJSON && (
          <Source id="selected-buildings" type="geojson" data={normalizedSelectedGeoJSON}>
            <Layer {...selectedFillLayer} />
            <Layer {...selectedOutlineLayer} />
          </Source>
        )}

        {selectionLocked && normalizedLockedSelectionGeoJSON && (
          <Source id="locked-selection-source" type="geojson" data={normalizedLockedSelectionGeoJSON}>
            <Layer {...lockedFillLayer} />
            <Layer {...lockedBuildingsLayer} />
            <Layer {...lockedOutlineLayer} />
          </Source>
        )}
      </Map>
    </div>
  );
}

export default MapView;

