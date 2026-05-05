import { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import ChatPanel from '../components/ChatPanel.jsx';
import RightPanel from '../components/RightPanel.jsx';
import SelectionPanel from '../components/SelectionPanel.jsx';
import {
  fetchBuildings,
  fetchConstructionTypeMapping,
} from '../services/api.js';
import {
  getFeatureStableKey,
  normalizeMapboxType,
  normalizeMapboxTypeList,
  parseGeoJSON,
} from '../utils/selection.js';

const MapView = lazy(() => import('../components/MapView.jsx'));

const USE_MAPBOX_BUILDINGS =
  (import.meta.env.VITE_USE_MAPBOX_BUILDINGS || 'true') === 'true';

// ── shared types ─────────────────────────────────────────────────────────────

type Row = Record<string, unknown>;

interface SelectionState {
  count: number;
  selectedGeoJSON: unknown;
  zipBase64: string | null;
  buildings: object[];
  selectionError: string | null;
}

interface ConstructionAreaState {
  count: number;
  selectedGeoJSON: unknown;
  buildings: object[];
  buildingKeys: string[];
  selectionError: string | null;
}

interface GeoFeature {
  properties?: Record<string, unknown>;
  [key: string]: unknown;
}

const INITIAL_SELECTION: SelectionState = {
  count: 0,
  selectedGeoJSON: null,
  zipBase64: null,
  buildings: [],
  selectionError: null,
};

const INITIAL_CONSTRUCTION_AREA_SELECTION: ConstructionAreaState = {
  count: 0,
  selectedGeoJSON: null,
  buildings: [],
  buildingKeys: [],
  selectionError: null,
};

// ── helpers ───────────────────────────────────────────────────────────────────

function getConstructionRowMapboxTypes(row: Row): string[] {
  const raw =
    row?.mapbox_type ?? row?.mapboxType ?? row?.mapbox_types ?? row?.mapboxTypes;
  return normalizeMapboxTypeList(raw);
}

function findBestConstructionRow(
  rows: Row[],
  useType: string,
  mapboxType: string,
  refurbishmentType: string,
  detail: string,
  yearStart: number,
  yearEnd: number,
): Row | null {
  const strictCandidates = rows.filter((row) => {
    const rowMapboxTypes = getConstructionRowMapboxTypes(row);
    const matchesMapboxType =
      !mapboxType ||
      !rowMapboxTypes.length ||
      rowMapboxTypes.includes(normalizeMapboxType(mapboxType));

    return (
      row?.cea_use_type1 === useType &&
      matchesMapboxType &&
      row?.refurbishment_type === refurbishmentType &&
      row?.detail === detail
    );
  });

  const candidates = strictCandidates.length
    ? strictCandidates
    : rows.filter(
        (row) =>
          row?.cea_use_type1 === useType &&
          row?.refurbishment_type === refurbishmentType &&
          row?.detail === detail,
      );

  if (!candidates.length) return null;

  const inclusive = candidates.find(
    (row) =>
      (row.year_start as number) <= yearStart &&
      (row.year_end as number) >= yearEnd,
  );
  if (inclusive) return inclusive;

  const overlapped = candidates
    .filter(
      (row) =>
        (row.year_end as number) >= yearStart &&
        (row.year_start as number) <= yearEnd,
    )
    .sort((a, b) => {
      const overlapA = Math.max(
        0,
        Math.min(a.year_end as number, yearEnd) -
          Math.max(a.year_start as number, yearStart),
      );
      const overlapB = Math.max(
        0,
        Math.min(b.year_end as number, yearEnd) -
          Math.max(b.year_start as number, yearStart),
      );
      return overlapB - overlapA;
    });

  return overlapped.length ? overlapped[0] : candidates[0];
}

// ── component ─────────────────────────────────────────────────────────────────

export interface ActiveSelectionInfo {
  selectedGeoJSON: unknown;
  count: number;
}

interface BuildingSelectionProps {
  onActiveSelectionChange: (info: ActiveSelectionInfo | null) => void;
  onDrawnPolygonChange: (polygon: unknown) => void;
}

function BuildingSelection({
  onActiveSelectionChange,
  onDrawnPolygonChange,
}: BuildingSelectionProps) {
  const [buildingsGeoJSON, setBuildingsGeoJSON] = useState<unknown>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [selection, setSelection] = useState<SelectionState>(INITIAL_SELECTION);
  const [confirmedSelection, setConfirmedSelection] = useState<SelectionState | null>(null);
  const [constructionMappingRows, setConstructionMappingRows] = useState<Row[]>([]);
  const [constructionMappingError, setConstructionMappingError] = useState<string | null>(null);
  const [constructionAreaSelection, setConstructionAreaSelection] =
    useState<ConstructionAreaState>(INITIAL_CONSTRUCTION_AREA_SELECTION);
  const [buildingAssignments, setBuildingAssignments] = useState<Record<string, Row>>({});
  const [drawnPolygon, setDrawnPolygon] = useState<unknown>(null);

  const handleSelection = useCallback((result: Row) => {
    setSelection({
      count: (result?.count as number) || 0,
      selectedGeoJSON: parseGeoJSON(result?.selected_geojson),
      zipBase64: (result?.zip_base64 as string) || null,
      buildings: Array.isArray(result?.buildings) ? (result.buildings as object[]) : [],
      selectionError: (result?.selection_error as string) || null,
    });
    if (result?.drawn_polygon) {
      setDrawnPolygon(result.drawn_polygon);
    }
  }, []);

  const handleConfirmSelection = useCallback(() => {
    if (!selection.count || !selection.selectedGeoJSON) return;
    setConfirmedSelection(selection);
    setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    setBuildingAssignments({});
  }, [selection]);

  const handleResetSelection = useCallback(() => {
    setConfirmedSelection(null);
    setSelection(INITIAL_SELECTION);
    setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    setBuildingAssignments({});
  }, []);

  const handleConstructionAreaSelection = useCallback((result: Row) => {
    setConstructionAreaSelection({
      count: (result?.count as number) || 0,
      selectedGeoJSON: parseGeoJSON(result?.selected_geojson),
      buildings: Array.isArray(result?.buildings) ? (result.buildings as object[]) : [],
      buildingKeys: Array.isArray(result?.building_keys)
        ? (result.building_keys as string[])
        : [],
      selectionError: (result?.selection_error as string) || null,
    });
  }, []);

  const handleConfirmConstructionFeatures = useCallback(
    ({
      useTypeSelections,
    }: {
      useTypeSelections: Record<string, Record<string, Row>>;
    }) => {
      if (
        !Array.isArray(constructionAreaSelection.buildingKeys) ||
        !constructionAreaSelection.buildingKeys.length
      ) {
        return;
      }

      const nextAssignments: Record<string, Row> = {};
      constructionAreaSelection.buildingKeys.forEach((key: string, index: number) => {
        const building = (constructionAreaSelection.buildings[index] || {}) as Row;
        const useType = String(building.cea_use_type1 || '').toUpperCase();
        const mapboxType = normalizeMapboxType(
          (building.mapbox_type ||
            building.type ||
            building.class ||
            building.building) as string,
        );
        const selected = useTypeSelections?.[useType]?.[mapboxType];
        if (
          !useType ||
          !mapboxType ||
          !selected?.refurbishment_type ||
          !selected?.detail ||
          !Number.isFinite(selected?.year_start) ||
          !Number.isFinite(selected?.year_end)
        ) {
          return;
        }

        const row = findBestConstructionRow(
          constructionMappingRows,
          useType,
          mapboxType,
          selected.refurbishment_type as string,
          selected.detail as string,
          selected.year_start as number,
          selected.year_end as number,
        );
        if (!row) return;

        nextAssignments[key] = {
          const_type: row.const_type,
          year_start: selected.year_start,
          year_end: selected.year_end,
          refurbishment_type: selected.refurbishment_type,
          detail: selected.detail,
          mapbox_type: mapboxType,
          cea_use_type1: useType,
        };
      });

      if (!Object.keys(nextAssignments).length) return;
      setBuildingAssignments((prev: Record<string, Row>) => ({
        ...prev,
        ...nextAssignments,
      }));
      setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    },
    [constructionAreaSelection, constructionMappingRows],
  );


  useEffect(() => {
    if (USE_MAPBOX_BUILDINGS) return;
    let cancelled = false;
    async function load() {
      try {
        const geojson = await fetchBuildings();
        if (!cancelled) setBuildingsGeoJSON(geojson);
      } catch (e) {
        if (!cancelled)
          setLoadError((e as Error)?.message || 'Failed to load buildings GeoJSON');
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const rows = await fetchConstructionTypeMapping();
        if (!cancelled) {
          const normalizedRows = Array.isArray(rows) ? (rows as Row[]) : [];
          setConstructionMappingRows(normalizedRows);
          setConstructionMappingError(
            normalizedRows.length
              ? null
              : 'Construction mapping is empty. Check CSV headers/content.',
          );
        }
      } catch (e) {
        if (!cancelled) {
          setConstructionMappingRows([]);
          setConstructionMappingError(
            (e as Error)?.message || 'Failed to load construction type mapping',
          );
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  // Report active selection upward so App.tsx can drive LeftDock and save/run actions.
  useEffect(() => {
    const target = confirmedSelection ?? selection;
    onActiveSelectionChange(
      target.count > 0 && target.selectedGeoJSON
        ? { selectedGeoJSON: target.selectedGeoJSON, count: target.count }
        : null,
    );
  }, [confirmedSelection, selection, onActiveSelectionChange]);

  useEffect(() => {
    onDrawnPolygonChange(drawnPolygon);
  }, [drawnPolygon, onDrawnPolygonChange]);

  // ── derived state ──────────────────────────────────────────────────────────

  const confirmedFeatureCollection = parseGeoJSON(
    confirmedSelection?.selectedGeoJSON,
  ) as { features?: GeoFeature[] } | null;

  const lockedSelectionGeoJSONWithState = (() => {
    if (!confirmedFeatureCollection?.features?.length) return null;
    const features = confirmedFeatureCollection.features;
    const total = features.length;
    const assignedCount = features.reduce((acc, feature, index) => {
      const key = getFeatureStableKey(feature, index);
      return acc + (buildingAssignments[key] ? 1 : 0);
    }, 0);
    const allAssigned = total > 0 && assignedCount === total;

    return {
      type: 'FeatureCollection',
      features: features.map((feature, index) => {
        const key = getFeatureStableKey(feature, index);
        const assignment = buildingAssignments[key] ?? null;
        const assignmentState = allAssigned
          ? 'complete'
          : assignment
            ? 'defined'
            : 'pending';
        return {
          ...feature,
          properties: {
            ...(feature?.properties || {}),
            __selection_key: key,
            __assignment_state: assignmentState,
            const_type: assignment?.const_type ?? null,
            refurbishment_type: assignment?.refurbishment_type ?? null,
            detail: assignment?.detail ?? null,
            feature_year_start: assignment?.year_start ?? null,
            feature_year_end: assignment?.year_end ?? null,
          },
        };
      }),
    };
  })();

  const confirmedBuildingsWithAssignments =
    (
      lockedSelectionGeoJSONWithState?.features as
        | Array<{ properties: Record<string, unknown> }>
        | undefined
    )?.map((f) => f.properties) ?? [];

  const definedBuildingCount = confirmedBuildingsWithAssignments.filter(
    (props) => props?.const_type,
  ).length;
  const allConstructionDefined =
    confirmedBuildingsWithAssignments.length > 0 &&
    definedBuildingCount === confirmedBuildingsWithAssignments.length;

  const activeSelection = confirmedSelection
    ? {
        ...confirmedSelection,
        count: confirmedBuildingsWithAssignments.length,
        buildings: confirmedBuildingsWithAssignments,
        selectedGeoJSON: lockedSelectionGeoJSONWithState,
      }
    : selection;

  // ── render ─────────────────────────────────────────────────────────────────

  return (
    <>
      <Suspense
        fallback={
          <div
            className="map-overlay"
            style={{ position: 'absolute', left: 12, top: 12 }}
          >
            Loading map...
          </div>
        }
      >
        <MapView
          buildingsGeoJSON={buildingsGeoJSON}
          selectedGeoJSON={selection.selectedGeoJSON}
          lockedSelectionGeoJSON={lockedSelectionGeoJSONWithState}
          selectionLocked={Boolean(confirmedSelection)}
          constructionPhaseActive={Boolean(confirmedSelection)}
          onSelection={handleSelection}
          onConstructionAreaSelection={handleConstructionAreaSelection}
          onDrawnPolygonChange={setDrawnPolygon}
        />
      </Suspense>

      {loadError && (
        <div
          className="map-overlay map-overlay-error"
          style={{
            position: 'absolute',
            left: 12,
            bottom: 12,
            padding: '8px 10px',
            borderRadius: 8,
            background: 'rgba(0,0,0,0.75)',
            color: 'white',
            fontSize: 12,
            maxWidth: 420,
          }}
        >
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
