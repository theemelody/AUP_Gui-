import { useCallback, useEffect, useMemo, useState } from 'react';
import { parseGeoJSON, getFeatureStableKey, normalizeMapboxType, normalizeMapboxTypeList } from '../utils/selection.js';
import type { ActiveSelectionInfo } from '../pages/buildingSelection';

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
  buildings: Row[];
  buildingKeys: string[];
  selectionError: string | null;
}

interface GeoFeature {
  properties?: Record<string, unknown>;
  [key: string]: unknown;
}

const INITIAL_SELECTION: SelectionState = {
  count: 0, selectedGeoJSON: null, zipBase64: null, buildings: [], selectionError: null,
};

const INITIAL_CONSTRUCTION_AREA_SELECTION: ConstructionAreaState = {
  count: 0, selectedGeoJSON: null, buildings: [], buildingKeys: [], selectionError: null,
};

// ── construction mapping helpers ──────────────────────────────────────────────

function getConstructionRowMapboxTypes(row: Row): string[] {
  const raw = row?.mapbox_type ?? row?.mapboxType ?? row?.mapbox_types ?? row?.mapboxTypes;
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
      (row.year_start as number) <= yearStart && (row.year_end as number) >= yearEnd,
  );
  if (inclusive) return inclusive;

  const overlapped = candidates
    .filter(
      (row) =>
        (row.year_end as number) >= yearStart && (row.year_start as number) <= yearEnd,
    )
    .sort((a, b) => {
      const overlapA = Math.max(0, Math.min(a.year_end as number, yearEnd) - Math.max(a.year_start as number, yearStart));
      const overlapB = Math.max(0, Math.min(b.year_end as number, yearEnd) - Math.max(b.year_start as number, yearStart));
      return overlapB - overlapA;
    });

  return overlapped.length ? overlapped[0] : candidates[0];
}

// ── hook ──────────────────────────────────────────────────────────────────────

interface UseBuildingSelectionOptions {
  loadedScenario: { geojson: unknown; drawnPolygon: unknown } | null | undefined;
  constructionMappingRows: Row[];
  onActiveSelectionChange: (info: ActiveSelectionInfo | null) => void;
  onDrawnPolygonChange: (polygon: unknown) => void;
  onConstTypesChange?: (types: string[]) => void;
  onBuildingCountsChange?: (counts: Record<string, number>) => void;
}

export function useBuildingSelection({
  loadedScenario,
  constructionMappingRows,
  onActiveSelectionChange,
  onDrawnPolygonChange,
  onConstTypesChange,
  onBuildingCountsChange,
}: UseBuildingSelectionOptions) {
  // Phase 1 → raw selection from map draw
  const [selection, setSelection] = useState<SelectionState>(INITIAL_SELECTION);
  // Phase 2 → user-confirmed selection (locks the map, enables construction assignment)
  const [confirmedSelection, setConfirmedSelection] = useState<SelectionState | null>(null);
  // Phase 3 → per-building construction type assignments
  const [buildingAssignments, setBuildingAssignments] = useState<Record<string, Row>>({});
  const [constructionAreaSelection, setConstructionAreaSelection] =
    useState<ConstructionAreaState>(INITIAL_CONSTRUCTION_AREA_SELECTION);
  const [drawnPolygon, setDrawnPolygon] = useState<unknown>(null);

  // Phase 1: accept raw result from MapView.onSelection
  const handleSelection = useCallback((result: Row) => {
    setSelection({
      count: (result?.count as number) || 0,
      selectedGeoJSON: parseGeoJSON(result?.selected_geojson),
      zipBase64: (result?.zip_base64 as string) || null,
      buildings: Array.isArray(result?.buildings) ? (result.buildings as object[]) : [],
      selectionError: (result?.selection_error as string) || null,
    });
    if (result?.drawn_polygon) setDrawnPolygon(result.drawn_polygon);
  }, []);

  // Phase 1→2: lock the map and start construction assignment
  const handleConfirmSelection = useCallback(() => {
    if (!selection.count || !selection.selectedGeoJSON) return;
    setConfirmedSelection(selection);
    setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    setBuildingAssignments({});
  }, [selection]);

  // Phase 2→1: unlock the map and clear all state
  const handleResetSelection = useCallback(() => {
    setConfirmedSelection(null);
    setSelection(INITIAL_SELECTION);
    setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    setBuildingAssignments({});
  }, []);

  // Phase 3: sub-selection within confirmed footprint for construction type assignment
  const handleConstructionAreaSelection = useCallback((result: Row) => {
    setConstructionAreaSelection({
      count: (result?.count as number) || 0,
      selectedGeoJSON: parseGeoJSON(result?.selected_geojson),
      buildings: Array.isArray(result?.buildings) ? (result.buildings as Row[]) : [],
      buildingKeys: Array.isArray(result?.building_keys) ? (result.building_keys as string[]) : [],
      selectionError: (result?.selection_error as string) || null,
    });
  }, []);

  const handleConfirmConstructionFeatures = useCallback(
    ({ useTypeSelections }: { useTypeSelections: Record<string, Record<string, Row>> }) => {
      if (!Array.isArray(constructionAreaSelection.buildingKeys) || !constructionAreaSelection.buildingKeys.length) return;

      const nextAssignments: Record<string, Row> = {};
      constructionAreaSelection.buildingKeys.forEach((key: string, index: number) => {
        const building = (constructionAreaSelection.buildings[index] || {}) as Row;
        const useType = String(building.cea_use_type1 || '').toUpperCase();
        const mapboxType = normalizeMapboxType(
          (building.mapbox_type || building.type || building.class || building.building) as string,
        );
        const selected = useTypeSelections?.[useType]?.[mapboxType];
        if (!useType || !mapboxType || !selected?.refurbishment_type || !selected?.detail ||
            !Number.isFinite(selected?.year_start) || !Number.isFinite(selected?.year_end)) return;

        const row = findBestConstructionRow(
          constructionMappingRows, useType, mapboxType,
          selected.refurbishment_type as string, selected.detail as string,
          selected.year_start as number, selected.year_end as number,
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
      setBuildingAssignments((prev) => ({ ...prev, ...nextAssignments }));
      setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    },
    [constructionAreaSelection, constructionMappingRows],
  );

  // Restore all state when a saved scenario is loaded/selected.
  useEffect(() => {
    if (!loadedScenario?.geojson) return;
    const fc = loadedScenario.geojson as { features?: GeoFeature[] };
    if (!Array.isArray(fc?.features) || !fc.features.length) return;

    const assignments: Record<string, Row> = {};
    fc.features.forEach((feature, index) => {
      const key = getFeatureStableKey(feature, index);
      const p = feature.properties || {};
      if (p.const_type) {
        assignments[key] = {
          const_type: p.const_type,
          refurbishment_type: p.refurbishment_type ?? null,
          detail: p.detail ?? null,
          year_start: p.feature_year_start ?? null,
          year_end: p.feature_year_end ?? null,
          mapbox_type: p.mapbox_type ?? null,
          cea_use_type1: p.cea_use_type1 ?? null,
        };
      }
    });

    const restoredSelection: SelectionState = {
      count: fc.features.length,
      selectedGeoJSON: loadedScenario.geojson,
      zipBase64: null,
      buildings: fc.features.map((f) => f.properties || {}),
      selectionError: null,
    };

    setSelection(restoredSelection);
    setConfirmedSelection(restoredSelection);
    setBuildingAssignments(assignments);
    if (loadedScenario.drawnPolygon) setDrawnPolygon(loadedScenario.drawnPolygon);
  }, [loadedScenario]);

  // ── derived state ─────────────────────────────────────────────────────────

  const confirmedFeatureCollection = useMemo(
    () => parseGeoJSON(confirmedSelection?.selectedGeoJSON) as { features?: GeoFeature[] } | null,
    [confirmedSelection],
  );

  const lockedSelectionGeoJSONWithState = useMemo(() => {
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
        const assignmentState = allAssigned ? 'complete' : assignment ? 'defined' : 'pending';
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
  }, [confirmedFeatureCollection, buildingAssignments]);

  const confirmedBuildingsWithAssignments = useMemo(
    () =>
      (
        lockedSelectionGeoJSONWithState?.features as
          | Array<{ properties: Record<string, unknown> }>
          | undefined
      )?.map((f) => f.properties) ?? [],
    [lockedSelectionGeoJSONWithState],
  );

  const definedBuildingCount = confirmedBuildingsWithAssignments.filter((p) => p?.const_type).length;
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

  // ── side-effects reporting upward ─────────────────────────────────────────

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

  useEffect(() => {
    const counts: Record<string, number> = {};
    for (const p of confirmedBuildingsWithAssignments) {
      const ct = p?.const_type as string;
      if (ct) counts[ct] = (counts[ct] ?? 0) + 1;
    }
    onConstTypesChange?.(Object.keys(counts));
    onBuildingCountsChange?.(counts);
  }, [confirmedBuildingsWithAssignments, onConstTypesChange, onBuildingCountsChange]);

  return {
    selection,
    confirmedSelection,
    constructionAreaSelection,
    drawnPolygon,
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
  };
}
