import { useCallback, useEffect, useRef } from "react";
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import type { MapRef } from "react-map-gl";
import {
  selectBuildingsFromMapbox,
  selectBuildingsFromFeatureCollection,
} from "../utils/mapbox.js";

interface UseMapboxDrawOptions {
  mapRef: React.RefObject<MapRef | null>;
  mapLoaded: boolean;
  selectionLocked: boolean;
  constructionPhaseActive: boolean;
  lockedSelectionGeoJSON: unknown;
  onSelection: ((result: Record<string, unknown>) => void) | undefined;
  onConstructionAreaSelection: ((result: Record<string, unknown>) => void) | undefined;
  onDrawnPolygonChange: ((polygon: unknown) => void) | undefined;
  ensureMappingLoaded: () => Promise<Record<string, string>>;
}

export function useMapboxDraw({
  mapRef,
  mapLoaded,
  selectionLocked,
  constructionPhaseActive,
  lockedSelectionGeoJSON,
  onSelection,
  onConstructionAreaSelection,
  onDrawnPolygonChange,
  ensureMappingLoaded,
}: UseMapboxDrawOptions) {
  const drawRef = useRef<MapboxDraw | null>(null);
  const selectDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Stable refs so the draw event handlers always see the latest props.
  const selectionLockedRef = useRef(selectionLocked);
  const constructionPhaseActiveRef = useRef(constructionPhaseActive);
  const lockedSelectionGeoJSONRef = useRef(lockedSelectionGeoJSON);
  const onSelectionRef = useRef(onSelection);
  const onConstructionAreaSelectionRef = useRef(onConstructionAreaSelection);
  const onDrawnPolygonChangeRef = useRef(onDrawnPolygonChange);

  useEffect(() => { selectionLockedRef.current = selectionLocked; }, [selectionLocked]);
  useEffect(() => { constructionPhaseActiveRef.current = constructionPhaseActive; }, [constructionPhaseActive]);
  useEffect(() => { lockedSelectionGeoJSONRef.current = lockedSelectionGeoJSON; }, [lockedSelectionGeoJSON]);
  useEffect(() => { onSelectionRef.current = onSelection; }, [onSelection]);
  useEffect(() => { onConstructionAreaSelectionRef.current = onConstructionAreaSelection; }, [onConstructionAreaSelection]);
  useEffect(() => { onDrawnPolygonChangeRef.current = onDrawnPolygonChange; }, [onDrawnPolygonChange]);

  const ensureDrawControl = useCallback(() => {
    if (!mapLoaded) return;
    const map = mapRef.current?.getMap?.();
    if (!map || drawRef.current) return;

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: { polygon: true, trash: true },
    });
    map.addControl(draw, "bottom-left");
    drawRef.current = draw;

    const fireSelection = (geometry: unknown) => {
      // Drawing is used for initial selection, then reused for construction-area selection.
      const isLocked = selectionLockedRef.current;
      const isConstructionPhase = constructionPhaseActiveRef.current;
      if (isLocked && !isConstructionPhase) return;

      if (selectDebounceRef.current) clearTimeout(selectDebounceRef.current);

      const fc = geometry as { features?: Array<{ geometry?: unknown }> } | null;
      const drawnPolygon = fc?.features?.[0]?.geometry ?? null;
      onDrawnPolygonChangeRef.current?.(drawnPolygon);

      selectDebounceRef.current = setTimeout(async () => {
        try {
          if (!geometry) {
            if (isLocked && isConstructionPhase) {
              onConstructionAreaSelectionRef.current?.({
                count: 0, selected_geojson: null, buildings: [], building_keys: [], selection_error: null,
              });
            } else {
              onSelectionRef.current?.({
                count: 0, selected_geojson: null, zip_base64: null, buildings: [], selection_error: null,
              });
            }
            return;
          }

          if (isLocked && isConstructionPhase) {
            const result = selectBuildingsFromFeatureCollection(
              lockedSelectionGeoJSONRef.current,
              geometry
            );
            onConstructionAreaSelectionRef.current?.({ ...result, selection_error: null });
          } else {
            const mapping = await ensureMappingLoaded();
            if (!mapping || Object.keys(mapping).length === 0) {
              throw new Error("Mapbox type mapping not loaded yet. Please try selection again.");
            }
            const result = selectBuildingsFromMapbox(map, geometry, mapping);
            onSelectionRef.current?.({ ...result, selection_error: null });
          }
        } catch (e: unknown) {
          if ((e as { name?: string })?.name === "AbortError") return;
          if (isLocked && isConstructionPhase) {
            onConstructionAreaSelectionRef.current?.({
              count: 0, selected_geojson: null, buildings: [], building_keys: [],
              selection_error: (e as Error)?.message || "Construction area selection failed",
            });
          } else {
            onSelectionRef.current?.({
              count: 0, selected_geojson: null, zip_base64: null, buildings: [],
              selection_error: (e as Error)?.message || "Selection failed",
            });
          }
        }
      }, 250);
    };

    const handleChange = () => {
      if (selectionLockedRef.current && !constructionPhaseActiveRef.current) return;
      const fc = draw.getAll();
      fireSelection(fc?.features?.length ? fc : null);
    };

    map.on("draw.create", handleChange);
    map.on("draw.update", handleChange);
    map.on("draw.delete", () => fireSelection(null));
  }, [mapRef, mapLoaded, ensureMappingLoaded]);

  // Initialize draw control and clean up debounce on unmount.
  useEffect(() => {
    ensureDrawControl();
    return () => {
      if (selectDebounceRef.current) clearTimeout(selectDebounceRef.current);
    };
  }, [ensureDrawControl]);

  // Show/hide draw UI buttons based on phase.
  useEffect(() => {
    const map = mapRef.current?.getMap?.();
    if (!map || !drawRef.current) return;

    const showDraw = !selectionLocked || constructionPhaseActive;
    const displayValue = showDraw ? "block" : "none";
    [
      ".mapbox-gl-draw_ctrl-draw-btn",
      ".mapbox-gl-draw_ctrl-draw-btn.mapbox-gl-draw_polygon",
      ".mapbox-gl-draw_trash",
    ].forEach((selector) => {
      map.getContainer().querySelectorAll(selector).forEach((el: Element) => {
        (el as HTMLElement).style.display = displayValue;
      });
    });
  }, [mapRef, constructionPhaseActive, selectionLocked]);

  // Clear stale draw shapes when switching between selection and construction phase.
  useEffect(() => {
    drawRef.current?.deleteAll();
  }, [selectionLocked]);

  return { drawRef };
}
