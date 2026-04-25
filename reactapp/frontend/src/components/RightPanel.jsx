import { useEffect, useMemo, useState } from "react";
import CollapsiblePanel from "./common/CollapsiblePanel.jsx";
import LabeledSelectField from "./common/LabeledSelectField.jsx";
import {
  getBuildingMapboxType,
  normalizeMapboxType,
  normalizeMapboxTypeList
} from "../utils/selection.js";

function uniqueNonEmpty(values) {
  return Array.from(
    new Set(values.map((value) => String(value || "").trim()).filter(Boolean))
  );
}

function normalizeUseType(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "");
}

function normalizeKey(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function getRowValue(row, aliases) {
  if (!row || typeof row !== "object") return "";

  const normalizedAliases = aliases.map((alias) => normalizeKey(alias));
  for (const [rawKey, rawValue] of Object.entries(row)) {
    const normalizedRawKey = normalizeKey(rawKey);
    if (normalizedAliases.includes(normalizedRawKey)) {
      return String(rawValue ?? "").trim();
    }
  }

  return "";
}

function getUseTypeValue(row) {
  return getRowValue(row, ["cea_use_type1", "ceaUseType1", "use_type", "useType"]);
}

function getRefurbishmentValue(row) {
  return getRowValue(row, [
    "refurbishment_type",
    "refurbishmentType",
    "refurbishment"
  ]);
}

function getDetailValue(row) {
  return getRowValue(row, ["detail", "detail_type", "detailType"]);
}

function getRowMapboxTypes(row) {
  const raw = row?.mapbox_type ?? row?.mapboxType ?? row?.mapbox_types ?? row?.mapboxTypes;
  return normalizeMapboxTypeList(raw);
}

function getYearRangeOptions(rows) {
  const options = [];
  const seen = new Set();

  rows.forEach((row) => {
    const yearStart = Number(row?.year_start);
    const yearEnd = Number(row?.year_end);
    if (!Number.isFinite(yearStart) || !Number.isFinite(yearEnd)) return;

    const value = `${yearStart}-${yearEnd}`;
    if (seen.has(value)) return;
    seen.add(value);

    options.push({
      value,
      label: `${yearStart}-${yearEnd}`,
      year_start: yearStart,
      year_end: yearEnd
    });
  });

  return options.sort((a, b) => a.year_start - b.year_start || a.year_end - b.year_end);
}

function toSelectOptions(values) {
  return values.map((value) => ({ value, label: value }));
}

function buildSelectionFromRows(rows, previousSelection = {}) {
  const refurbOptions = uniqueNonEmpty(rows.map((row) => getRefurbishmentValue(row)));
  const refurbishment_type = refurbOptions.includes(previousSelection.refurbishment_type)
    ? previousSelection.refurbishment_type
    : refurbOptions[0] || "";

  const detailSourceRows = refurbishment_type
    ? rows.filter((row) => getRefurbishmentValue(row) === refurbishment_type)
    : rows;
  const detailOptions = uniqueNonEmpty(detailSourceRows.map((row) => getDetailValue(row)));
  const detail = detailOptions.includes(previousSelection.detail)
    ? previousSelection.detail
    : detailOptions[0] || "";

  const yearRangeSourceRows =
    refurbishment_type && detail
      ? rows.filter(
          (row) =>
            getRefurbishmentValue(row) === refurbishment_type &&
            getDetailValue(row) === detail
        )
      : [];
  const yearRangeOptions = getYearRangeOptions(yearRangeSourceRows);
  const selectedYearRange =
    yearRangeOptions.find((option) => option.value === previousSelection.year_range) ||
    yearRangeOptions[0] ||
    null;

  return {
    refurbishment_type,
    detail,
    year_range: selectedYearRange?.value || "",
    year_start: selectedYearRange?.year_start ?? null,
    year_end: selectedYearRange?.year_end ?? null,
    hasOptions: Boolean(refurbOptions.length && detailOptions.length && yearRangeOptions.length)
  };
}

function RightPanel({
  rightCollapsed,
  setRightCollapsed,
  constructionPhaseActive,
  mappingRows,
  mappingError,
  constructionAreaSelection,
  onConfirmConstructionFeatures,
  totalConfirmedBuildings,
  definedBuildingCount,
  allConstructionDefined
}) {
  const [useTypeSelections, setUseTypeSelections] = useState({});
  const [expandedUseType, setExpandedUseType] = useState("");
  const [expandedMapboxPanel, setExpandedMapboxPanel] = useState("");

  const areaUseTypes = useMemo(() => {
    const useTypes = (constructionAreaSelection?.buildings || [])
      .map((building) => String(building?.cea_use_type1 || "").trim().toUpperCase())
      .filter(Boolean);
    return Array.from(new Set(useTypes)).sort();
  }, [constructionAreaSelection]);

  const areaBuildingsByUseType = useMemo(() => {
    const grouped = {};
    (constructionAreaSelection?.buildings || []).forEach((building) => {
      const useType = String(building?.cea_use_type1 || "").trim().toUpperCase();
      if (!useType) return;
      if (!grouped[useType]) grouped[useType] = [];
      grouped[useType].push(building);
    });
    return grouped;
  }, [constructionAreaSelection]);

  const areaBuildingsByUseTypeAndMapbox = useMemo(() => {
    const grouped = {};
    (constructionAreaSelection?.buildings || []).forEach((building) => {
      const useType = String(building?.cea_use_type1 || "").trim().toUpperCase();
      const mapboxType = getBuildingMapboxType(building);
      if (!useType || !mapboxType) return;
      if (!grouped[useType]) grouped[useType] = {};
      if (!grouped[useType][mapboxType]) grouped[useType][mapboxType] = [];
      grouped[useType][mapboxType].push(building);
    });

    return grouped;
  }, [constructionAreaSelection]);

  const areaMapboxTypesByUseType = useMemo(() => {
    const grouped = {};
    Object.entries(areaBuildingsByUseTypeAndMapbox).forEach(([useType, byMapbox]) => {
      grouped[useType] = Object.keys(byMapbox).sort();
    });
    return grouped;
  }, [areaBuildingsByUseTypeAndMapbox]);

  const rowsByUseType = useMemo(() => {
    const exact = {};
    const normalized = {};
    const normalizedAreaSet = new Set(areaUseTypes.map((useType) => normalizeUseType(useType)));

    for (const row of mappingRows || []) {
      const useType = String(getUseTypeValue(row) || "").trim().toUpperCase();
      const normalizedUseType = normalizeUseType(useType);
      if (!normalizedAreaSet.has(normalizedUseType)) continue;

      if (!exact[useType]) exact[useType] = [];
      exact[useType].push(row);

      if (!normalized[normalizedUseType]) normalized[normalizedUseType] = [];
      normalized[normalizedUseType].push(row);
    }

    return { exact, normalized };
  }, [areaUseTypes, mappingRows]);

  const getRowsForUseType = (useType) => {
    const exactRows = rowsByUseType.exact?.[useType] || [];
    if (exactRows.length) return exactRows;
    return rowsByUseType.normalized?.[normalizeUseType(useType)] || [];
  };

  const getRowsForUseTypeAndMapboxType = (useType, mapboxType) =>
    getFilteredRowsForUseType(useType, mapboxType);

  const getMapboxTypeOptionsForUseType = (useType) =>
    uniqueNonEmpty((areaBuildingsByUseType?.[useType] || []).map((building) => getBuildingMapboxType(building)));

  const getFilteredRowsForUseType = (useType, mapboxType) => {
    const useRows = getRowsForUseType(useType);
    const normalizedMapboxType = normalizeMapboxType(mapboxType);
    if (!normalizedMapboxType) return useRows;

    const filteredRows = useRows.filter((row) => {
      const rowMapboxTypes = getRowMapboxTypes(row);
      return !rowMapboxTypes.length || rowMapboxTypes.includes(normalizedMapboxType);
    });

    // If no row is tagged for this specific mapbox type, fallback to use-type rows.
    return filteredRows.length ? filteredRows : useRows;
  };

  useEffect(() => {
    if (!areaUseTypes.length) {
      setUseTypeSelections({});
      return;
    }

    setUseTypeSelections((prev) => {
      const next = {};

      areaUseTypes.forEach((useType) => {
        next[useType] = {};
        const mapboxTypes = areaMapboxTypesByUseType[useType] || [];

        mapboxTypes.forEach((mapboxType) => {
          const previousSelection = prev?.[useType]?.[mapboxType] || {};
          const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
          next[useType][mapboxType] = {
            ...buildSelectionFromRows(useRows, previousSelection),
            mapbox_type: mapboxType
          };
        });
      });

      return next;
    });
  }, [areaMapboxTypesByUseType, areaUseTypes, rowsByUseType]);

  useEffect(() => {
    if (!areaUseTypes.length) {
      setExpandedUseType("");
      setExpandedMapboxPanel("");
      return;
    }

    setExpandedUseType((prev) =>
      prev && areaUseTypes.includes(prev) ? prev : areaUseTypes[0]
    );
    setExpandedMapboxPanel((prev) => {
      const availablePanels = areaUseTypes.flatMap((useType) =>
        (areaMapboxTypesByUseType[useType] || []).map((mapboxType) => `${useType}::${mapboxType}`)
      );
      return prev && availablePanels.includes(prev) ? prev : availablePanels[0] || "";
    });
  }, [areaMapboxTypesByUseType, areaUseTypes]);

  const getSelectionFor = (useType, mapboxType) =>
    useTypeSelections?.[useType]?.[mapboxType] || null;

  const canConfirmFeatures =
    constructionAreaSelection?.count > 0 &&
    areaUseTypes.length > 0 &&
    areaUseTypes.every((useType) => {
      const mapboxTypes = areaMapboxTypesByUseType[useType] || [];
      return mapboxTypes.every((mapboxType) => {
        const current = useTypeSelections?.[useType]?.[mapboxType];
        return Boolean(
          current?.mapbox_type &&
            current?.refurbishment_type &&
            current?.detail &&
            current?.year_range
        );
      });
    });

  const handleSelectionChange = (useType, mapboxType, nextSelection) => {
    setUseTypeSelections((prev) => ({
      ...prev,
      [useType]: {
        ...(prev?.[useType] || {}),
        [mapboxType]: {
          ...(prev?.[useType]?.[mapboxType] || {}),
          ...nextSelection,
          mapbox_type: mapboxType
        }
      }
    }));
  };

  const handleRefurbishmentChange = (useType, mapboxType, refurbishment_type) => {
    const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
    const detailOptions = uniqueNonEmpty(
      useRows
        .filter((row) => getRefurbishmentValue(row) === refurbishment_type)
        .map((row) => getDetailValue(row))
    );
    const nextDetail = detailOptions[0] || "";
    const yearRangeSourceRows = nextDetail
      ? useRows.filter(
          (row) =>
            getRefurbishmentValue(row) === refurbishment_type &&
            getDetailValue(row) === nextDetail
        )
      : [];
    const yearRangeOptions = getYearRangeOptions(yearRangeSourceRows);
    const selectedYearRange = yearRangeOptions[0] || null;

    handleSelectionChange(useType, mapboxType, {
      refurbishment_type,
      detail: nextDetail,
      year_range: selectedYearRange?.value || "",
      year_start: selectedYearRange?.year_start ?? null,
      year_end: selectedYearRange?.year_end ?? null
    });
  };

  const handleDetailChange = (useType, mapboxType, detail) => {
    const currentRefurbishment = useTypeSelections?.[useType]?.[mapboxType]?.refurbishment_type || "";
    const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
    const yearRangeSourceRows = currentRefurbishment
      ? useRows.filter(
          (row) =>
            getRefurbishmentValue(row) === currentRefurbishment &&
            getDetailValue(row) === detail
        )
      : [];
    const yearRangeOptions = getYearRangeOptions(yearRangeSourceRows);
    const selectedYearRange = yearRangeOptions[0] || null;

    handleSelectionChange(useType, mapboxType, {
      detail,
      year_range: selectedYearRange?.value || "",
      year_start: selectedYearRange?.year_start ?? null,
      year_end: selectedYearRange?.year_end ?? null
    });
  };

  const handleYearRangeChange = (useType, mapboxType, yearRange) => {
    const current = useTypeSelections?.[useType]?.[mapboxType] || {};
    const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
    const yearRangeSourceRows =
      current.refurbishment_type && current.detail
        ? useRows.filter(
            (row) =>
              getRefurbishmentValue(row) === current.refurbishment_type &&
              getDetailValue(row) === current.detail
          )
        : [];
    const yearRangeOptions = getYearRangeOptions(yearRangeSourceRows);
    const selectedYearRange =
      yearRangeOptions.find((option) => option.value === yearRange) || null;

    handleSelectionChange(useType, mapboxType, {
      year_range: selectedYearRange?.value || "",
      year_start: selectedYearRange?.year_start ?? null,
      year_end: selectedYearRange?.year_end ?? null
    });
  };

  const handleConfirm = () => {
    if (!canConfirmFeatures) return;
    onConfirmConstructionFeatures?.({
      useTypeSelections
    });
  };

  return (
    <CollapsiblePanel
      positionClass="bottom-panel-right"
      collapsed={rightCollapsed}
      setCollapsed={setRightCollapsed}
      title="Construction type phase"
      expandAriaLabel="Expand right panel"
      collapseAriaLabel="Collapse right panel"
    >
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
              <div className={allConstructionDefined ? "status-complete" : "status-pending"}>
                {allConstructionDefined
                  ? "All defined"
                  : "In progress"}
              </div>
            </div>

            {mappingError && (
              <div className="construction-error">Mapping error: {mappingError}</div>
            )}

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
                Draw an area covering preselected buildings to see filtered refurbishment
                and detail options.
              </div>
            ) : (
              <div className="construction-use-panels">
                {areaUseTypes.map((useType) => {
                  const isExpanded = expandedUseType === useType;
                  const mapboxTypes = areaMapboxTypesByUseType[useType] || [];

                  return (
                    <div
                      className={[
                        "construction-section",
                        "construction-type-accordion",
                        isExpanded ? "is-expanded" : ""
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      key={useType}
                    >
                      <button
                        type="button"
                        className="construction-type-toggle"
                        aria-expanded={isExpanded}
                        aria-controls={`construction-panel-${useType}`}
                        onClick={() => setExpandedUseType(isExpanded ? "" : useType)}
                      >
                        <span className="construction-use-type">Use type: {useType}</span>
                        <span className="construction-type-caret">{isExpanded ? "▲" : "▼"}</span>
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
                              const selection = getSelectionFor(useType, mapboxType) || {};
                              const useRows = getRowsForUseTypeAndMapboxType(useType, mapboxType);
                              const effectiveRows = useRows.length ? useRows : getRowsForUseType(useType);
                              const refurbOptions = uniqueNonEmpty(
                                effectiveRows.map((row) => getRefurbishmentValue(row))
                              );
                              const refurbSelectOptions = toSelectOptions(refurbOptions);
                              const detailOptions = uniqueNonEmpty(
                                (selection.refurbishment_type
                                  ? effectiveRows.filter(
                                      (row) =>
                                        getRefurbishmentValue(row) === selection.refurbishment_type
                                    )
                                  : effectiveRows
                                ).map((row) => getDetailValue(row))
                              );
                              const detailSelectOptions = toSelectOptions(detailOptions);
                              const yearRangeOptions = getYearRangeOptions(
                                selection.refurbishment_type && selection.detail
                                  ? effectiveRows.filter(
                                      (row) =>
                                        getRefurbishmentValue(row) === selection.refurbishment_type &&
                                        getDetailValue(row) === selection.detail
                                    )
                                  : []
                              );
                              const canShowBody = isMapboxExpanded;
                              const fallbackNotice = !useRows.length
                                ? "No mapbox-specific rows found; showing use-type defaults."
                                : null;

                              return (
                                <div
                                  key={panelKey}
                                  className={[
                                    "construction-subsection",
                                    isMapboxExpanded ? "is-expanded" : ""
                                  ]
                                    .filter(Boolean)
                                    .join(" ")}
                                >
                                  <button
                                    type="button"
                                    className="construction-subsection-toggle"
                                    aria-expanded={isMapboxExpanded}
                                    aria-controls={`construction-subpanel-${panelKey}`}
                                    onClick={() =>
                                      setExpandedMapboxPanel(
                                        isMapboxExpanded ? "" : panelKey
                                      )
                                    }
                                  >
                                    <span className="construction-subsection-title">
                                      {mapboxType}
                                    </span>
                                    <span className="construction-type-caret">
                                      {isMapboxExpanded ? "▲" : "▼"}
                                    </span>
                                  </button>

                                  {canShowBody && (
                                    <div
                                      id={`construction-subpanel-${panelKey}`}
                                      className="construction-subsection-body"
                                    >
                                      {fallbackNotice && (
                                        <div className="construction-muted construction-fallback-note">
                                          {fallbackNotice}
                                        </div>
                                      )}

                                      <LabeledSelectField
                                        id={`refurb-${panelKey}`}
                                        label="Refurbishment type"
                                        value={selection.refurbishment_type || ""}
                                        onChange={(e) =>
                                          handleRefurbishmentChange(
                                            useType,
                                            mapboxType,
                                            e.target.value
                                          )
                                        }
                                        options={refurbSelectOptions}
                                        emptyLabel="No refurbishment options"
                                      />

                                      <LabeledSelectField
                                        id={`detail-${panelKey}`}
                                        label="Detail type"
                                        value={selection.detail || ""}
                                        onChange={(e) =>
                                          handleDetailChange(useType, mapboxType, e.target.value)
                                        }
                                        options={detailSelectOptions}
                                        emptyLabel="No detail options"
                                      />

                                      <LabeledSelectField
                                        id={`year-range-${panelKey}`}
                                        label="Year range"
                                        value={selection.year_range || ""}
                                        onChange={(e) =>
                                          handleYearRangeChange(
                                            useType,
                                            mapboxType,
                                            e.target.value
                                          )
                                        }
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
    </CollapsiblePanel>
  );
}

export default RightPanel;
