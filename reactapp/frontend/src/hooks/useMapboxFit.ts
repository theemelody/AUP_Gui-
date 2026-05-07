import { useCallback, useEffect } from "react";
import type { MapRef } from "react-map-gl";
import { getBoundsFromGeoJSON } from "../utils/mapbox.js";

interface UseMapboxFitOptions {
  mapRef: React.RefObject<MapRef | null>;
  /** When non-null and mapLoaded is true, the map flies to this GeoJSON (e.g. a restored scenario). */
  fitToGeoJSON: unknown;
  mapLoaded: boolean;
}

export function useMapboxFit({ mapRef, fitToGeoJSON, mapLoaded }: UseMapboxFitOptions) {
  const fitToBounds = useCallback(
    (geojson: unknown, options?: object) => {
      const bounds = getBoundsFromGeoJSON(geojson);
      if (!bounds) return;
      const map = mapRef.current?.getMap?.();
      map?.fitBounds(bounds as [[number, number], [number, number]], {
        padding: 60,
        maxZoom: 18,
        ...options,
      });
    },
    [mapRef]
  );

  // Fly to a loaded/restored scenario when it becomes available.
  useEffect(() => {
    if (!fitToGeoJSON || !mapLoaded) return;
    fitToBounds(fitToGeoJSON, { padding: 80, maxZoom: 17, duration: 1200 });
  }, [fitToGeoJSON, mapLoaded, fitToBounds]);

  return { fitToBounds };
}
