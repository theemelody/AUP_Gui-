import { useEffect, useMemo, useState } from 'react';
import { Collapse } from 'react-collapse';
import LabeledSelectField from './common/LabeledSelectField.jsx';
import { getBuildingMapboxType, normalizeMapboxType, normalizeMapboxTypeList } from '../utils/selection.js';

type Row = Record<string, unknown>;

interface YearRangeOption {
  value: string;
  label: string;
  year_start: number;
  year_end: number;
}

interface PanelSelection {
  [key: string]: unknown;
  refurbishment_type: string;
  detail: string;
  year_range: string;
  year_start: number | null;
  year_end: number | null;
  mapbox_type?: string;
  hasOptions?: boolean;
}

type UseTypeSelections = Record<string, Record<string, PanelSelection>>;

interface ConstructionAreaSelection {
  count: number;
  buildings: Row[];
  selectionError?: string | null;
}

interface RightPanelProps {
  rightCollapsed: boolean;
  setRightCollapsed: (v: boolean) => void;
  constructionPhaseActive: boolean;
  mappingRows: Row[];
  mappingError: string | null;
  constructionAreaSelection: ConstructionAreaSelection;
  onConfirmConstructionFeatures: (args: { useTypeSelections: UseTypeSelections }) => void;
  totalConfirmedBuildings: number;
  definedBuildingCount: number;
  allConstructionDefined: boolean;
}

// ── pure helpers ──────────────────────────────────────────────────────────────

function uniqueNonEmpty(values: unknown[]): string[] {
  return Array.from(new Set(values.map((v) => String(v || '').trim()).filter(Boolean)));
}

function normalizeUseType(value: unknown): string {
  return String(value || '').trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
}

function normalizeKey(value: unknown): string {
  return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
}

function getRowValue(row: Row, aliases: string[]): string {
  if (!row || typeof row !== 'object') return '';
  const normalizedAliases = aliases.map(normalizeKey);
  for (const [rawKey, rawValue] of Object.entries(row)) {
    if (normalizedAliases.includes(normalizeKey(rawKey))) return String(rawValue ?? '').trim();
  }
  return '';
}

const getUseTypeValue = (row: Row) => getRowValue(row, ['cea_use_type1', 'ceaUseType1', 'use_type', 'useType']);
const getRefurbishmentValue = (row: Row) => getRowValue(row, ['refurbishment_type', 'refurbishmentType', 'refurbishment']);
const getDetailValue = (row: Row) => getRowValue(row, ['detail', 'detail_type', 'detailType']);

function getRowMapboxTypes(row: Row): string[] {
  const raw = row?.mapbox_type ?? row?.mapboxType ?? row?.mapbox_types ?? row?.mapboxTypes;
  return normalizeMapboxTypeList(raw);
}

function getYearRangeOptions(rows: Row[]): YearRangeOption[] {
  const options: YearRangeOption[] = [];
  const seen = new Set<string>();

  rows.forEach((row) => {
    const yearStart = Number(row?.year_start);
    const yearEnd = Number(row?.year_end);
    if (!Number.isFinite(yearStart) || !Number.isFinite(yearEnd)) return;
    const value = `${yearStart}-${yearEnd}`;
    if (seen.has(value)) return;
    seen.add(value);
    options.push({ value, label: value, year_start: yearStart, year_end: yearEnd });
  });

  return options.sort((a, b) => a.year_start - b.year_start || a.year_end - b.year_end);
}

function toSelectOptions(values: string[]): { value: string; label: string }[] {
  return values.map((value) => ({ value, label: value }));
}

function buildSelectionFromRows(rows: Row[], previousSelection: Partial<PanelSelection> = {}): PanelSelection {
  const refurbOptions = uniqueNonEmpty(rows.map(getRefurbishmentValue));
  const refurbishment_type = refurbOptions.includes(previousSelection.refurbishment_type ?? '')
    ? previousSelection.refurbishment_type!
    : refurbOptions[0] || '';

  const detailSourceRows = refurbishment_type
    ? rows.filter((row) => getRefurbishmentValue(row) === refurbishment_type)
    : rows;
  const detailOptions = uniqueNonEmpty(detailSourceRows.map(getDetailValue));
  const detail = detailOptions.includes(previousSelection.detail ?? '')
    ? previousSelection.detail!
    : detailOptions[0] || '';

  const yearRangeSourceRows =
    refurbishment_type && detail
      ? rows.filter((row) => getRefurbishmentValue(row) === refurbishment_type && getDetailValue(row) === detail)
      : [];
  const yearRangeOptions = getYearRangeOptions(yearRangeSourceRows);
  const selectedYearRange =
    yearRangeOptions.find((o) => o.value === previousSelection.year_range) || yearRangeOptions[0] || null;

  return {
    refurbishment_type,
    detail,
    year_range: selectedYearRange?.value || '',
    year_start: selectedYearRange?.year_start ?? null,
    year_end: selectedYearRange?.year_end ?? null,
    hasOptions: Boolean(refurbOptions.length && detailOptions.length && yearRangeOptions.length),
  };
}

// ── component ─────────────────────────────────────────────────────────────────

function RightPanel({
  rightCollapsed, setRightCollapsed, constructionPhaseActive,
  mappingRows, mappingError, constructionAreaSelection,
  onConfirmConstructionFeatures, totalConfirmedBuildings, definedBuildingCount, allConstructionDefined,
}: RightPanelProps) {
  const [useTypeSelections, setUseTypeSelections] = useState<UseTypeSelections>({});
  const [expandedUseType, setExpandedUseType] = useState('');
  const [expandedMapboxPanel, setExpandedMapboxPanel] = useState('');

  const areaUseTypes = useMemo(() => {
    const useTypes = (constructionAreaSelection?.buildings || [])
      .map((b) => String(b?.cea_use_type1 || '').trim().toUpperCase())
      .filter(Boolean);
    return Array.from(new Set(useTypes)).sort();
  }, [constructionAreaSelection]);

  const areaBuildingsByUseTypeAndMapbox = useMemo(() => {
    const grouped: Record<string, Record<string, Row[]>> = {};
    (constructionAreaSelection?.buildings || []).forEach((b) => {
      const useType = String(b?.cea_use_type1 || '').trim().toUpperCase();
      const mapboxType = getBuildingMapboxType(b);
      if (!useType || !mapboxType) return;
      (grouped[useType] ??= {})[mapboxType] ??= [];
      grouped[useType][mapboxType].push(b);
    });
    return grouped;
  }, [constructionAreaSelection]);

  const areaMapboxTypesByUseType = useMemo(() => {
    const grouped: Record<string, string[]> = {};
    Object.entries(areaBuildingsByUseTypeAndMapbox).forEach(([useType, byMapbox]) => {
      grouped[useType] = Object.keys(byMapbox).sort();
    });
    return grouped;
  }, [areaBuildingsByUseTypeAndMapbox]);

  const rowsByUseType = useMemo(() => {
    const exact: Record<string, Row[]> = {};
    const normalized: Record<string, Row[]> = {};
    const normalizedAreaSet = new Set(areaUseTypes.map(normalizeUseType));

    for (const row of mappingRows || []) {
      const useType = String(getUseTypeValue(row) || '').trim().toUpperCase();
      const normalizedUseType = normalizeUseType(useType);
      if (!normalizedAreaSet.has(normalizedUseType)) continue;
      (exact[useType] ??= []).push(row);
      (normalized[normalizedUseType] ??= []).push(row);
    }

    return { exact, normalized };
  }, [areaUseTypes, mappingRows]);

  const getRowsForUseType = (useType: string): Row[] => {
    const exactRows = rowsByUseType.exact?.[useType] || [];
    return exactRows.length ? exactRows : (rowsByUseType.normalized?.[normalizeUseType(useType)] || []);
  };

  const getFilteredRowsForUseType = (useType: string, mapboxType: string): Row[] => {
    const useRows = getRowsForUseType(useType);
    const normalizedMapboxType = normalizeMapboxType(mapboxType);
    if (!normalizedMapboxType) return useRows;
    const filteredRows = useRows.filter((row) => {
      const rowMapboxTypes = getRowMapboxTypes(row);
      // If no row is tagged for this specific mapbox type, fallback to use-type rows.
      return !rowMapboxTypes.length || rowMapboxTypes.includes(normalizedMapboxType);
    });
    return filteredRows.length ? filteredRows : useRows;
  };

  const getRowsForUseTypeAndMapboxType = (useType: string, mapboxType: string): Row[] =>
    getFilteredRowsForUseType(useType, mapboxType);

  useEffect(() => {
    if (!areaUseTypes.length) { setUseTypeSelections({}); return; }

    setUseTypeSelections((prev) => {
      const next: UseTypeSelections = {};
      areaUseTypes.forEach((useType) => {
        next[useType] = {};
        (areaMapboxTypesByUseType[useType] || []).forEach((mapboxType) => {
          const previousSelection = prev?.[useType]?.[mapboxType] || {};
          const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
          next[useType][mapboxType] = { ...buildSelectionFromRows(useRows, previousSelection), mapbox_type: mapboxType };
        });
      });
      return next;
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [areaMapboxTypesByUseType, areaUseTypes, rowsByUseType]);

  useEffect(() => {
    if (!areaUseTypes.length) { setExpandedUseType(''); setExpandedMapboxPanel(''); return; }
    setExpandedUseType((prev) => (prev && areaUseTypes.includes(prev) ? prev : areaUseTypes[0]));
    setExpandedMapboxPanel((prev) => {
      const availablePanels = areaUseTypes.flatMap((ut) =>
        (areaMapboxTypesByUseType[ut] || []).map((mt) => `${ut}::${mt}`)
      );
      return prev && availablePanels.includes(prev) ? prev : availablePanels[0] || '';
    });
  }, [areaMapboxTypesByUseType, areaUseTypes]);

  const getSelectionFor = (useType: string, mapboxType: string): PanelSelection | null =>
    useTypeSelections?.[useType]?.[mapboxType] || null;

  const canConfirmFeatures =
    constructionAreaSelection?.count > 0 &&
    areaUseTypes.length > 0 &&
    areaUseTypes.every((useType) =>
      (areaMapboxTypesByUseType[useType] || []).every((mapboxType) => {
        const current = useTypeSelections?.[useType]?.[mapboxType];
        return Boolean(current?.mapbox_type && current?.refurbishment_type && current?.detail && current?.year_range);
      })
    );

  const handleCascadeChange = (useType: string, mapboxType: string, field: string, value: string) => {
    const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
    const current = useTypeSelections?.[useType]?.[mapboxType] || ({} as PanelSelection);

    let nextRefurbishment = current.refurbishment_type;
    let nextDetail = current.detail;

    if (field === 'refurbishment_type') {
      nextRefurbishment = value;
      nextDetail = uniqueNonEmpty(
        useRows.filter((row) => getRefurbishmentValue(row) === value).map(getDetailValue)
      )[0] || '';
    } else if (field === 'detail') {
      nextDetail = value;
    }

    const yearRangeSourceRows =
      nextRefurbishment && nextDetail
        ? useRows.filter((row) => getRefurbishmentValue(row) === nextRefurbishment && getDetailValue(row) === nextDetail)
        : [];
    const yearRangeOptions = getYearRangeOptions(yearRangeSourceRows);
    const selectedYearRange =
      field === 'year_range' ? (yearRangeOptions.find((o) => o.value === value) || null) : (yearRangeOptions[0] || null);

    setUseTypeSelections((prev) => ({
      ...prev,
      [useType]: {
        ...(prev?.[useType] || {}),
        [mapboxType]: {
          ...(prev?.[useType]?.[mapboxType] || ({} as PanelSelection)),
          refurbishment_type: nextRefurbishment,
          detail: nextDetail,
          year_range: selectedYearRange?.value || '',
          year_start: selectedYearRange?.year_start ?? null,
          year_end: selectedYearRange?.year_end ?? null,
          mapbox_type: mapboxType,
        },
      },
    }));
  };

  const handleConfirm = () => {
    if (!canConfirmFeatures) return;
    onConfirmConstructionFeatures?.({ useTypeSelections });
  };

  return (
    <section className="bottom-panel-right" aria-label="Construction type phase">
      <div>
        <button type="button" className="action-link" onClick={() => setRightCollapsed(!rightCollapsed)}>
          {rightCollapsed ? 'Show Construction type phase' : 'Hide Construction type phase'}
        </button>
      </div>
      <Collapse isOpened={!rightCollapsed}>
        <div>
          {!constructionPhaseActive ? (
            <div className="construction-muted">
              Confirm building selection first. Confirmed buildings will turn orange, then
              you can draw areas and define their construction features.
            </div>
          ) : (
            <>
              <div className="construction-status">
                <div>Confirmed buildings: {totalConfirmedBuildings}</div>
                <div>Defined: {definedBuildingCount}</div>
                <div className={allConstructionDefined ? 'status-complete' : 'status-pending'}>
                  {allConstructionDefined ? 'All defined' : 'In progress'}
                </div>
              </div>

              {mappingError && <div className="construction-error">Mapping error: {mappingError}</div>}

              <div className="construction-section">
                <div className="construction-label">Area selection</div>
                <div className="construction-muted">
                  Draw an area over orange buildings. Selected in current area: {constructionAreaSelection?.count || 0}
                </div>
                {constructionAreaSelection?.selectionError && (
                  <div className="construction-error">
                    Selection error: {constructionAreaSelection.selectionError}
                  </div>
                )}
              </div>

              {areaUseTypes.length === 0 ? (
                <div className="construction-muted">
                  Draw an area covering preselected buildings to see filtered refurbishment and detail options.
                </div>
              ) : (
                <div className="construction-use-panels">
                  {areaUseTypes.map((useType) => {
                    const isExpanded = expandedUseType === useType;
                    const mapboxTypes = areaMapboxTypesByUseType[useType] || [];

                    return (
                      <div
                        key={useType}
                        className={['construction-section', 'construction-type-accordion', isExpanded ? 'is-expanded' : ''].filter(Boolean).join(' ')}
                      >
                        <button
                          type="button"
                          className="construction-type-toggle"
                          aria-expanded={isExpanded}
                          aria-controls={`construction-panel-${useType}`}
                          onClick={() => setExpandedUseType(isExpanded ? '' : useType)}
                        >
                          <span className="construction-use-type">Use type: {useType}</span>
                          <span className="construction-type-caret">{isExpanded ? '▲' : '▼'}</span>
                        </button>

                        {isExpanded && (
                          <div id={`construction-panel-${useType}`} className="construction-type-body">
                            <div className="construction-muted construction-mapbox-hint">
                              {mapboxTypes.length === 1
                                ? `Mapbox type: ${mapboxTypes[0]}`
                                : `${mapboxTypes.length} mapbox types in this use type`}
                            </div>

                            <div className="construction-mapbox-panels">
                              {mapboxTypes.map((mapboxType) => {
                                const panelKey = `${useType}::${mapboxType}`;
                                const isMapboxExpanded = expandedMapboxPanel === panelKey;
                                const sel = getSelectionFor(useType, mapboxType) || ({} as PanelSelection);
                                const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
                                const effectiveRows = useRows.length ? useRows : getRowsForUseType(useType);
                                const refurbSelectOptions = toSelectOptions(uniqueNonEmpty(effectiveRows.map(getRefurbishmentValue)));
                                const detailSelectOptions = toSelectOptions(uniqueNonEmpty(
                                  (sel.refurbishment_type
                                    ? effectiveRows.filter((row) => getRefurbishmentValue(row) === sel.refurbishment_type)
                                    : effectiveRows
                                  ).map(getDetailValue)
                                ));
                                const yearRangeOptions = getYearRangeOptions(
                                  sel.refurbishment_type && sel.detail
                                    ? effectiveRows.filter((row) =>
                                        getRefurbishmentValue(row) === sel.refurbishment_type && getDetailValue(row) === sel.detail
                                      )
                                    : []
                                );

                                return (
                                  <div
                                    key={panelKey}
                                    className={['construction-subsection', isMapboxExpanded ? 'is-expanded' : ''].filter(Boolean).join(' ')}
                                  >
                                    <button
                                      type="button"
                                      className="construction-subsection-toggle"
                                      aria-expanded={isMapboxExpanded}
                                      aria-controls={`construction-subpanel-${panelKey}`}
                                      onClick={() => setExpandedMapboxPanel(isMapboxExpanded ? '' : panelKey)}
                                    >
                                      <span className="construction-subsection-title">{mapboxType}</span>
                                      <span className="construction-type-caret">{isMapboxExpanded ? '▲' : '▼'}</span>
                                    </button>

                                    {isMapboxExpanded && (
                                      <div id={`construction-subpanel-${panelKey}`} className="construction-subsection-body">
                                        {!useRows.length && (
                                          <div className="construction-muted construction-fallback-note">
                                            No mapbox-specific rows found; showing use-type defaults.
                                          </div>
                                        )}
                                        <LabeledSelectField
                                          id={`refurb-${panelKey}`}
                                          label="Refurbishment type"
                                          value={sel.refurbishment_type || ''}
                                          onChange={(e) => handleCascadeChange(useType, mapboxType, 'refurbishment_type', e.target.value)}
                                          options={refurbSelectOptions}
                                          emptyLabel="No refurbishment options"
                                        />
                                        <LabeledSelectField
                                          id={`detail-${panelKey}`}
                                          label="Detail type"
                                          value={sel.detail || ''}
                                          onChange={(e) => handleCascadeChange(useType, mapboxType, 'detail', e.target.value)}
                                          options={detailSelectOptions}
                                          emptyLabel="No detail options"
                                        />
                                        <LabeledSelectField
                                          id={`year-range-${panelKey}`}
                                          label="Year range"
                                          value={sel.year_range || ''}
                                          onChange={(e) => handleCascadeChange(useType, mapboxType, 'year_range', e.target.value)}
                                          options={yearRangeOptions}
                                          emptyLabel="No year range options"
                                        />
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              <button
                type="button"
                className="action-button construction-confirm-btn"
                onClick={handleConfirm}
                disabled={!canConfirmFeatures}
              >
                Confirm features
              </button>
            </>
          )}
        </div>
      </Collapse>
    </section>
  );
}

export default RightPanel;
