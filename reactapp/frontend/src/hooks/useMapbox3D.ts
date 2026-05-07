import { useCallback, useRef } from "react";
import type { MapRef } from "react-map-gl";

interface UseMapbox3DOptions {
  mapRef: React.RefObject<MapRef | null>;
  selectionLocked: boolean;
}

export function useMapbox3D({ mapRef, selectionLocked }: UseMapbox3DOptions) {
  const has3DInitRef = useRef(false);

  const init3D = useCallback(() => {
    const map = mapRef.current?.getMap?.();
    if (!map || has3DInitRef.current) return;

    if (!map.getSource("mapbox-dem")) {
      map.addSource("mapbox-dem", {
        type: "raster-dem",
        url: "mapbox://mapbox.mapbox-terrain-dem-v1",
        tileSize: 512,
        maxzoom: 14,
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
          "sky-atmosphere-sun-intensity": 12,
        },
      });
    }

    if (!map.getLayer("3d-buildings") && map.getSource("composite")) {
      const labelLayerId = map
        .getStyle()
        ?.layers?.find((l: { type: string; layout?: Record<string, unknown> }) => l.type === "symbol" && l.layout?.["text-field"])
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
            "fill-extrusion-base": ["get", "min_height"],
          },
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
  }, [mapRef, selectionLocked]);

  return { init3D };
}
