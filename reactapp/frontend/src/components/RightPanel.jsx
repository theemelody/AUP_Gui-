import { useEffect, useMemo, useState } from "react";

const MIN_YEAR = 1800;
const MAX_YEAR = 2030;

function clampYear(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return MIN_YEAR;
  return Math.max(MIN_YEAR, Math.min(MAX_YEAR, Math.round(numeric)));
}

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
  const [yearStart, setYearStart] = useState(MIN_YEAR);
  const [yearEnd, setYearEnd] = useState(MAX_YEAR);
  const [useTypeSelections, setUseTypeSelections] = useState({});

  const areaUseTypes = useMemo(() => {
    const useTypes = (constructionAreaSelection?.buildings || [])
      .map((building) => String(building?.cea_use_type1 || "").trim().toUpperCase())
      .filter(Boolean);
    return Array.from(new Set(useTypes)).sort();
  }, [constructionAreaSelection]);

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

  useEffect(() => {
    if (!areaUseTypes.length) {
      setUseTypeSelections({});
      return;
    }

    setUseTypeSelections((prev) => {
      const next = {};

      areaUseTypes.forEach((useType) => {
        const useRows = getRowsForUseType(useType);
        const refurbOptions = uniqueNonEmpty(
          useRows.map((row) => getRefurbishmentValue(row))
        );

        const previousRefurb = prev?.[useType]?.refurbishment_type;
        const refurbishment_type = refurbOptions.includes(previousRefurb)
          ? previousRefurb
          : refurbOptions[0] || "";

        const detailSourceRows = refurbishment_type
          ? useRows.filter((row) => getRefurbishmentValue(row) === refurbishment_type)
          : useRows;

        const detailOptions = uniqueNonEmpty(
          detailSourceRows.map((row) => getDetailValue(row))
        );

        const previousDetail = prev?.[useType]?.detail;
        const detail = detailOptions.includes(previousDetail)
          ? previousDetail
          : detailOptions[0] || "";

        next[useType] = {
          refurbishment_type,
          detail
        };
      });

      return next;
    });
  }, [areaUseTypes, rowsByUseType]);

  const canConfirmFeatures =
    constructionAreaSelection?.count > 0 &&
    areaUseTypes.length > 0 &&
    areaUseTypes.every(
      (useType) =>
        useTypeSelections?.[useType]?.refurbishment_type &&
        useTypeSelections?.[useType]?.detail
    );

  const handleRefurbishmentChange = (useType, refurbishment_type) => {
    setUseTypeSelections((prev) => {
      const useRows = getRowsForUseType(useType);
      const detailOptions = uniqueNonEmpty(
        useRows
          .filter((row) => getRefurbishmentValue(row) === refurbishment_type)
          .map((row) => getDetailValue(row))
      );

      return {
        ...prev,
        [useType]: {
          refurbishment_type,
          detail: detailOptions[0] || ""
        }
      };
    });
  };

  const handleDetailChange = (useType, detail) => {
    setUseTypeSelections((prev) => ({
      ...prev,
      [useType]: {
        ...(prev?.[useType] || {}),
        detail
      }
    }));
  };

  const handleConfirm = () => {
    if (!canConfirmFeatures) return;
    onConfirmConstructionFeatures?.({
      yearStart,
      yearEnd,
      useTypeSelections
    });
  };

  return (
    <aside
      className={[
        "bottom-panel",
        "bottom-panel-right",
        rightCollapsed ? "is-collapsed" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="bottom-panel-header">
        <div className="bottom-panel-title">Construction type phase</div>
        <button
          type="button"
          className="bottom-panel-toggle"
          aria-label={rightCollapsed ? "Expand right panel" : "Collapse right panel"}
          onClick={() => setRightCollapsed((v) => !v)}
        >
          {rightCollapsed ? "▲" : "▼"}
        </button>
      </div>
      <div className="bottom-panel-body">
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
                  ? "All defined: buildings are green"
                  : "In progress: defined buildings are yellow"}
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

            <div className="construction-section">
              <div className="construction-label">
                Year range ({yearStart} - {yearEnd})
              </div>
              <div className="construction-range-group">
                <label className="construction-range-label" htmlFor="year-start-range">
                  Start year
                </label>
                <input
                  id="year-start-range"
                  type="range"
                  min={MIN_YEAR}
                  max={MAX_YEAR}
                  value={yearStart}
                  onChange={(e) => {
                    const next = clampYear(e.target.value);
                    setYearStart(Math.min(next, yearEnd));
                  }}
                />
              </div>
              <div className="construction-range-group">
                <label className="construction-range-label" htmlFor="year-end-range">
                  End year
                </label>
                <input
                  id="year-end-range"
                  type="range"
                  min={MIN_YEAR}
                  max={MAX_YEAR}
                  value={yearEnd}
                  onChange={(e) => {
                    const next = clampYear(e.target.value);
                    setYearEnd(Math.max(next, yearStart));
                  }}
                />
              </div>
            </div>

            {areaUseTypes.length === 0 ? (
              <div className="construction-muted">
                Draw an area covering preselected buildings to see filtered refurbishment
                and detail options.
              </div>
            ) : (
              areaUseTypes.map((useType) => {
                const useRows = getRowsForUseType(useType);
                const refurbOptions = uniqueNonEmpty(
                  useRows.map((row) => getRefurbishmentValue(row))
                );
                const selectedRefurb =
                  useTypeSelections?.[useType]?.refurbishment_type || "";
                const detailOptions = uniqueNonEmpty(
                  (selectedRefurb
                    ? useRows.filter(
                        (row) => getRefurbishmentValue(row) === selectedRefurb
                      )
                    : useRows
                  ).map((row) => getDetailValue(row))
                );
                const selectedDetail = useTypeSelections?.[useType]?.detail || "";

                return (
                  <div className="construction-section" key={useType}>
                    <div className="construction-use-type">Use type: {useType}</div>

                    <label
                      className="construction-field-label"
                      htmlFor={`refurb-${useType}`}
                    >
                      Refurbishment type
                    </label>
                    <select
                      id={`refurb-${useType}`}
                      className="construction-select"
                      value={selectedRefurb || ""}
                      onChange={(e) =>
                        handleRefurbishmentChange(useType, e.target.value)
                      }
                      disabled={!refurbOptions.length}
                    >
                      {!refurbOptions.length && (
                        <option value="">No refurbishment options</option>
                      )}
                      {refurbOptions.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>

                    <label
                      className="construction-field-label"
                      htmlFor={`detail-${useType}`}
                    >
                      Detail type
                    </label>
                    <select
                      id={`detail-${useType}`}
                      className="construction-select"
                      value={selectedDetail || ""}
                      onChange={(e) => handleDetailChange(useType, e.target.value)}
                      disabled={!detailOptions.length}
                    >
                      {!detailOptions.length && (
                        <option value="">No detail options</option>
                      )}
                      {detailOptions.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })
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
    </aside>
  );
}

export default RightPanel;
