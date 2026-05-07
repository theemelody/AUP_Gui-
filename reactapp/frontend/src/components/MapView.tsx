import { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import MapComponent, {
  FullscreenControl,
  NavigationControl,
  Source,
  Layer,
} from 'react-map-gl';
import { parseGeoJSON } from '../utils/selection.js';
import {
  buildingsFillLayer,
  buildingsOutlineLayer,
  selectedFillLayer,
  selectedOutlineLayer,
  lockedBuildingsLayer,
  lockedFillLayer,
  lockedOutlineLayer,
} from '../utils/mapbox.js';
import { useMapboxMapping } from '../hooks/useMapboxMapping';
import { useMapbox3D } from '../hooks/useMapbox3D';
import { useMapboxDraw } from '../hooks/useMapboxDraw';
import { useMapboxFit } from '../hooks/useMapboxFit';
import type { ViewState, LayerProps } from 'react-map-gl';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_ACCESS_TOKEN as string;

interface MapViewProps {
  buildingsGeoJSON?: unknown;
  selectedGeoJSON?: unknown;
  lockedSelectionGeoJSON?: unknown;
  selectionLocked: boolean;
  constructionPhaseActive: boolean;
  onSelection: (result: Record<string, unknown>) => void;
  onConstructionAreaSelection: (result: Record<string, unknown>) => void;
  onDrawnPolygonChange: (polygon: unknown) => void;
  fitToGeoJSON?: unknown;
}

function MapView({
  buildingsGeoJSON,
  selectedGeoJSON,
  lockedSelectionGeoJSON,
  selectionLocked,
  constructionPhaseActive,
  onSelection,
  onConstructionAreaSelection,
  onDrawnPolygonChange,
  fitToGeoJSON,
}: MapViewProps) {
  const mapRef = useRef<import('react-map-gl').MapRef | null>(null);
  const [viewState, setViewState] = useState<Partial<ViewState>>({
    longitude: 2.1734,
    latitude: 41.3851,
    zoom: 12.5,
    pitch: 60,
    bearing: -20,
  });
  const [mapLoaded, setMapLoaded] = useState(false);

  const geojson = useMemo(() => parseGeoJSON(buildingsGeoJSON), [buildingsGeoJSON]);
  const normalizedSelectedGeoJSON = useMemo(() => parseGeoJSON(selectedGeoJSON), [selectedGeoJSON]);
  const normalizedLockedSelectionGeoJSON = useMemo(() => parseGeoJSON(lockedSelectionGeoJSON), [lockedSelectionGeoJSON]);

  const hasFittedRef = useRef(false);

  const { ensureMappingLoaded } = useMapboxMapping();
  const { init3D } = useMapbox3D({ mapRef, selectionLocked });
  const { fitToBounds } = useMapboxFit({ mapRef, fitToGeoJSON, mapLoaded });

  useMapboxDraw({
    mapRef,
    mapLoaded,
    selectionLocked,
    constructionPhaseActive,
    lockedSelectionGeoJSON: normalizedLockedSelectionGeoJSON,
    onSelection,
    onConstructionAreaSelection,
    onDrawnPolygonChange,
    ensureMappingLoaded,
  });

  // For SHP mode: fit to static buildings extent when geojson loads or map loads, whichever is last.
  useEffect(() => {
    if (!geojson || !mapLoaded || hasFittedRef.current) return;
    fitToBounds(geojson);
    hasFittedRef.current = true;
  }, [geojson, mapLoaded, fitToBounds]);

  const handleMapLoad = useCallback(() => {
    init3D();
    setMapLoaded(true);
  }, [init3D]);

  // Hide basemap 3D buildings once the user's selection is confirmed.
  useEffect(() => {
    const map = mapRef.current?.getMap?.();
    const layer = map?.getLayer('3d-buildings');
    if (layer) {
      map!.setLayoutProperty('3d-buildings', 'visibility', selectionLocked ? 'none' : 'visible');
    }
  }, [selectionLocked]);

  const savedPitchRef = useRef<number | null>(null);

  // Flatten pitch during construction phase so draw vertices align with the ground plane.
  useEffect(() => {
    if (!mapLoaded) return;
    const map = mapRef.current?.getMap?.();
    if (!map) return;

    if (constructionPhaseActive) {
      if (savedPitchRef.current === null) savedPitchRef.current = map.getPitch();
      map.easeTo({ pitch: 0, duration: 180 });
      return;
    }

    if (savedPitchRef.current !== null) {
      map.easeTo({ pitch: savedPitchRef.current, duration: 180 });
      savedPitchRef.current = null;
    }
  }, [constructionPhaseActive, mapLoaded]);

  // Keep map sized to its container via ResizeObserver.
  useEffect(() => {
    if (!mapLoaded) return;
    const map = mapRef.current?.getMap?.();
    const container = map?.getContainer();
    if (!container || typeof ResizeObserver === 'undefined') return;

    map!.resize();
    const observer = new ResizeObserver(() => map!.resize());
    observer.observe(container);
    return () => observer.disconnect();
  }, [mapLoaded]);

  return (
    <div className="map-container">
      <MapComponent
        ref={mapRef}
        {...viewState}
        onLoad={handleMapLoad}
        onMove={(evt) => setViewState(evt.viewState)}
        mapStyle="mapbox://styles/mapbox/light-v11"
        mapboxAccessToken={MAPBOX_TOKEN}
      >
        <NavigationControl position="bottom-left" />
        <FullscreenControl position="bottom-left" />

        {geojson && (
          <Source id="buildings" type="geojson" data={geojson as object}>
            <Layer {...(buildingsFillLayer as LayerProps)} />
            <Layer {...(buildingsOutlineLayer as LayerProps)} />
          </Source>
        )}

        {!selectionLocked && normalizedSelectedGeoJSON && (
          <Source id="selected-buildings" type="geojson" data={normalizedSelectedGeoJSON as object}>
            <Layer {...(selectedFillLayer as LayerProps)} />
            <Layer {...(selectedOutlineLayer as LayerProps)} />
          </Source>
        )}

        {selectionLocked && normalizedLockedSelectionGeoJSON && (
          <Source id="locked-selection-source" type="geojson" data={normalizedLockedSelectionGeoJSON as object}>
            <Layer {...(lockedFillLayer as LayerProps)} />
            <Layer {...(lockedBuildingsLayer as LayerProps)} />
            <Layer {...(lockedOutlineLayer as LayerProps)} />
          </Source>
        )}
      </MapComponent>
    </div>
  );
}

export default MapView;
