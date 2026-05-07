import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import ChatPanel from '../components/ChatPanel.jsx';
import RightPanel from '../components/RightPanel.jsx';
import SelectionPanel from '../components/SelectionPanel.jsx';
import { fetchBuildings, fetchConstructionTypeMapping } from '../services/api.js';
import { useBuildingSelection } from '../hooks/useBuildingSelection';

const MapView = lazy(() => import('../components/MapView.jsx'));

const USE_MAPBOX_BUILDINGS =
  (import.meta.env.VITE_USE_MAPBOX_BUILDINGS || 'true') === 'true';

export interface ActiveSelectionInfo {
  selectedGeoJSON: unknown;
  count: number;
}

interface BuildingSelectionProps {
  onActiveSelectionChange: (info: ActiveSelectionInfo | null) => void;
  onDrawnPolygonChange: (polygon: unknown) => void;
  onConstTypesChange?: (types: string[]) => void;
  onBuildingCountsChange?: (counts: Record<string, number>) => void;
  loadedScenario?: { geojson: unknown; drawnPolygon: unknown } | null;
}

function BuildingSelection({
  onActiveSelectionChange,
  onDrawnPolygonChange,
  onConstTypesChange,
  onBuildingCountsChange,
  loadedScenario,
}: BuildingSelectionProps) {
  const [buildingsGeoJSON, setBuildingsGeoJSON] = useState<unknown>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [constructionMappingRows, setConstructionMappingRows] = useState<Record<string, unknown>[]>([]);
  const [constructionMappingError, setConstructionMappingError] = useState<string | null>(null);

  useEffect(() => {
    if (USE_MAPBOX_BUILDINGS) return;
    let cancelled = false;
    fetchBuildings()
      .then((geojson) => { if (!cancelled) setBuildingsGeoJSON(geojson); })
      .catch((e: Error) => { if (!cancelled) setLoadError(e?.message || 'Failed to load buildings GeoJSON'); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchConstructionTypeMapping()
      .then((rows: unknown) => {
        if (cancelled) return;
        const normalized = Array.isArray(rows) ? rows as Record<string, unknown>[] : [];
        setConstructionMappingRows(normalized);
        setConstructionMappingError(normalized.length ? null : 'Construction mapping is empty. Check CSV headers/content.');
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setConstructionMappingRows([]);
        setConstructionMappingError(e?.message || 'Failed to load construction type mapping');
      });
    return () => { cancelled = true; };
  }, []);

  const stableOnDrawnPolygonChange = useCallback(
    (polygon: unknown) => onDrawnPolygonChange(polygon),
    [onDrawnPolygonChange],
  );

  const {
    selection,
    confirmedSelection,
    constructionAreaSelection,
    activeSelection,
    lockedSelectionGeoJSONWithState,
    confirmedBuildingsWithAssignments,
    definedBuildingCount,
    allConstructionDefined,
    handleSelection,
    handleConfirmSelection,
    handleResetSelection,
    handleConstructionAreaSelection,
    handleConfirmConstructionFeatures,
    setDrawnPolygon,
  } = useBuildingSelection({
    loadedScenario,
    constructionMappingRows,
    onActiveSelectionChange,
    onDrawnPolygonChange: stableOnDrawnPolygonChange,
    onConstTypesChange,
    onBuildingCountsChange,
  });

  return (
    <>
      <Suspense fallback={<div className="map-overlay map-overlay-top-left">Loading map...</div>}>
        <MapView
          buildingsGeoJSON={buildingsGeoJSON}
          selectedGeoJSON={selection.selectedGeoJSON}
          lockedSelectionGeoJSON={lockedSelectionGeoJSONWithState}
          selectionLocked={Boolean(confirmedSelection)}
          constructionPhaseActive={Boolean(confirmedSelection)}
          onSelection={handleSelection}
          onConstructionAreaSelection={handleConstructionAreaSelection}
          onDrawnPolygonChange={setDrawnPolygon}
          fitToGeoJSON={confirmedSelection?.selectedGeoJSON}
        />
      </Suspense>

      {loadError && (
        <div className="map-overlay map-overlay-error">
          Buildings load error: {loadError}
        </div>
      )}

      <ChatPanel
        leftCollapsed={leftCollapsed}
        setLeftCollapsed={setLeftCollapsed}
        activeSelectionCount={activeSelection.count}
      />

      <SelectionPanel
        selection={selection}
        confirmedSelection={confirmedSelection}
        activeSelection={activeSelection}
        handleConfirmSelection={handleConfirmSelection}
        handleResetSelection={handleResetSelection}
      />

      <RightPanel
        rightCollapsed={rightCollapsed}
        setRightCollapsed={setRightCollapsed}
        constructionPhaseActive={Boolean(confirmedSelection)}
        mappingRows={constructionMappingRows}
        mappingError={constructionMappingError}
        constructionAreaSelection={constructionAreaSelection}
        onConfirmConstructionFeatures={handleConfirmConstructionFeatures}
        totalConfirmedBuildings={confirmedBuildingsWithAssignments.length}
        definedBuildingCount={definedBuildingCount}
        allConstructionDefined={allConstructionDefined}
      />
    </>
  );
}

export default BuildingSelection;
