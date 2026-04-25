const BUILDING_KEY_PRIORITY = [
  "const_type",
  "refurbishment_type",
  "detail",
  "mapbox_type",
  "feature_year_start",
  "feature_year_end",
  "cea_use_type1",
  "type",
  "height",
  "min_height",
  "estimated_floors"
];

function getOrderedBuildingEntries(building) {
  // Keep most relevant building fields at the top of the card for easier review.
  const entries = Object.entries(building || {});
  const priority = new Map(BUILDING_KEY_PRIORITY.map((key, idx) => [key, idx]));

  return entries.sort(([keyA], [keyB]) => {
    const rankA = priority.has(keyA) ? priority.get(keyA) : Number.MAX_SAFE_INTEGER;
    const rankB = priority.has(keyB) ? priority.get(keyB) : Number.MAX_SAFE_INTEGER;
    if (rankA !== rankB) return rankA - rankB;
    return keyA.localeCompare(keyB);
  });
}

function SelectionPanel({
  selection,
  confirmedSelection,
  activeSelection,
  handleConfirmSelection,
  handleResetSelection
}) {
  return (
    <section className="selected-list-panel" aria-label="Selected buildings">
      <div className="selected-list-header">
        <div className="selected-list-title">
          {confirmedSelection
            ? `Confirmed Buildings (${confirmedSelection.count})`
            : `Selected Buildings (${selection.count})`}
        </div>
        <div className="selected-list-actions">
          {confirmedSelection ? (
            <button type="button" className="action-link" onClick={handleResetSelection}>
              Reset selection
            </button>
          ) : (
            <button
              type="button"
              className="action-link"
              onClick={handleConfirmSelection}
              disabled={!selection.count || !selection.selectedGeoJSON}
            >
              Confirm selection
            </button>
          )}
        </div>
      </div>

      <div className="selected-list-body">
        {selection.selectionError && !confirmedSelection && (
          <div className="selected-empty selected-error">
            Selection error: {selection.selectionError}
          </div>
        )}
        {activeSelection.buildings.length === 0 ? (
          <div className="selected-empty">
            No buildings selected yet.
          </div>
        ) : (
          activeSelection.buildings.map((building, index) => (
            <article className="building-card" key={index}>
              <div className="building-card-title">Building {index + 1}</div>
              <div className="building-props">
                {getOrderedBuildingEntries(building).map(([key, value]) => (
                  <div className="building-prop-row" key={key}>
                    <span className="building-prop-key">{key}</span>
                    <span className="building-prop-value">
                      {value === null || value === undefined
                        ? "-"
                        : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}

export default SelectionPanel;
