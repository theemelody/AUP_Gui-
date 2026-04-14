import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import Map, {
  FullscreenControl,
  NavigationControl,
  Source,
  Layer
} from "react-map-gl";
import MapboxDraw from "@mapbox/mapbox-gl-draw";

import { selectBuildings } from "../services/api.js";

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_ACCESS_TOKEN;

function normalizeGeoJSON(value) {
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

function MapView({ buildingsGeoJSON, selectedGeoJSON, onSelection }) {
  const mapRef = useRef(null);
  const [viewState, setViewState] = useState({
    longitude: 2.1734,
    latitude: 41.3851,
    zoom: 12.5,
    pitch: 60,
    bearing: -20
  });

  const geojson = useMemo(
    () => normalizeGeoJSON(buildingsGeoJSON),
    [buildingsGeoJSON]
  );
  const normalizedSelectedGeoJSON = useMemo(
    () => normalizeGeoJSON(selectedGeoJSON),
    [selectedGeoJSON]
  );
  const hasFittedRef = useRef(false);
  const drawRef = useRef(null);
  const selectAbortRef = useRef(null);
  const selectDebounceRef = useRef(null);

  const fitMapToBounds = useCallback(() => {
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
    }

    has3DInitRef.current = true;
  }, []);

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
      if (selectDebounceRef.current) clearTimeout(selectDebounceRef.current);
      selectDebounceRef.current = setTimeout(async () => {
        try {
          if (!geometry) {
            onSelection?.({
              count: 0,
              selected_geojson: null,
              zip_base64: null,
              buildings: [],
              selection_error: null
            });
            return;
          }
          if (selectAbortRef.current) selectAbortRef.current.abort();
          const ctrl = new AbortController();
          selectAbortRef.current = ctrl;
          const result = await selectBuildings(geometry, { signal: ctrl.signal });
          onSelection?.({ ...result, selection_error: null });
        } catch (e) {
          if (e?.name === "AbortError") return;
          onSelection?.({
            count: 0,
            selected_geojson: null,
            zip_base64: null,
            buildings: [],
            selection_error: e?.message || "Selection failed"
          });
        }
      }, 250);
    };

    const handleChange = () => {
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
  }, [onSelection]);

  const handleMapLoad = useCallback(() => {
    init3D();
    ensureDrawControl();
    if (geojson && !hasFittedRef.current) fitMapToBounds();
  }, [ensureDrawControl, geojson, fitMapToBounds, init3D]);

  useEffect(() => {
    ensureDrawControl();
    return () => {
      if (selectDebounceRef.current) clearTimeout(selectDebounceRef.current);
      if (selectAbortRef.current) selectAbortRef.current.abort();
    };
  }, [ensureDrawControl]);

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

        {normalizedSelectedGeoJSON && (
          <Source id="selected-buildings" type="geojson" data={normalizedSelectedGeoJSON}>
            <Layer {...selectedFillLayer} />
            <Layer {...selectedOutlineLayer} />
          </Source>
        )}
      </Map>
    </div>
  );
}

export default MapView;

