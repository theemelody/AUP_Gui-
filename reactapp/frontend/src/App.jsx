import { useCallback, useEffect, useState } from "react";
import MapView from "./components/MapView.jsx";
import LeftDock from "./components/LeftDock.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import SelectionPanel from "./components/SelectionPanel.jsx";
import RightPanel from "./components/RightPanel.jsx";
import {
  fetchBuildings,
  fetchConstructionTypeMapping,
  sendChatMessage
} from "./services/api.js";

const USE_MAPBOX_BUILDINGS =
  (import.meta.env.VITE_USE_MAPBOX_BUILDINGS || "true") === "true";

const INITIAL_SELECTION = {
  count: 0,
  selectedGeoJSON: null,
  zipBase64: null,
  buildings: [],
  selectionError: null
};

const INITIAL_CONSTRUCTION_AREA_SELECTION = {
  count: 0,
  selectedGeoJSON: null,
  buildings: [],
  buildingKeys: [],
  selectionError: null
};

function parseSelectedGeoJSON(value) {
  if (!value) return null;
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function getFeatureStableKey(feature, index = 0) {
  const props = feature?.properties || {};
  const explicitKey = props.__selection_key;
  if (explicitKey) return String(explicitKey);

  const candidates = [
    feature?.id,
    props.id,
    props.osm_id,
    props.osm_way_id,
    props.mapbox_id
  ];

  for (const value of candidates) {
    if (value !== null && value !== undefined && value !== "") {
      return String(value);
    }
  }

  return `geom-${index}-${JSON.stringify(feature?.geometry || {})}`;
}

function normalizeMapboxType(value) {
  return String(value || "").trim().toLowerCase();
}

function getConstructionRowMapboxTypes(row) {
  const raw = row?.mapbox_type ?? row?.mapboxType ?? row?.mapbox_types ?? row?.mapboxTypes;
  const values = Array.isArray(raw)
    ? raw
    : String(raw || "")
        .split(/[,;|]/)
        .map((item) => item.trim())
        .filter(Boolean);

  return Array.from(
    new Set(values.map((value) => normalizeMapboxType(value)).filter(Boolean))
  );
}

function findBestConstructionRow(
  rows,
  useType,
  mapboxType,
  refurbishmentType,
  detail,
  yearStart,
  yearEnd
) {
  const strictCandidates = rows.filter(
    (row) => {
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
    }
  );
  const candidates = strictCandidates.length
    ? strictCandidates
    : rows.filter(
        (row) =>
          row?.cea_use_type1 === useType &&
          row?.refurbishment_type === refurbishmentType &&
          row?.detail === detail
      );
  if (!candidates.length) return null;

  const inclusive = candidates.find(
    (row) => row.year_start <= yearStart && row.year_end >= yearEnd
  );
  if (inclusive) return inclusive;

  const overlapped = candidates
    .filter((row) => row.year_end >= yearStart && row.year_start <= yearEnd)
    .sort((a, b) => {
      const overlapA = Math.max(0, Math.min(a.year_end, yearEnd) - Math.max(a.year_start, yearStart));
      const overlapB = Math.max(0, Math.min(b.year_end, yearEnd) - Math.max(b.year_start, yearStart));
      return overlapB - overlapA;
    });

  if (overlapped.length) return overlapped[0];
  return candidates[0];
}

function App() {
  const [buildingsGeoJSON, setBuildingsGeoJSON] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text: "Hi! I am connected to Ollama. Ask me anything about your selected buildings."
    }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);
  const [scenarioName, setScenarioName] = useState("");
  const [savedScenarios, setSavedScenarios] = useState([
    "munich-commercial-scenario"
  ]);
  const [selection, setSelection] = useState(INITIAL_SELECTION);
  const [confirmedSelection, setConfirmedSelection] = useState(null);
  const [constructionMappingRows, setConstructionMappingRows] = useState([]);
  const [constructionMappingError, setConstructionMappingError] = useState(null);
  const [constructionAreaSelection, setConstructionAreaSelection] = useState(
    INITIAL_CONSTRUCTION_AREA_SELECTION
  );
  const [buildingAssignments, setBuildingAssignments] = useState({});

  const handleSelection = useCallback((result) => {
    // Normalize backend and frontend selection payloads into one app state shape.
    setSelection({
      count: result?.count || 0,
      selectedGeoJSON: parseSelectedGeoJSON(result?.selected_geojson),
      zipBase64: result?.zip_base64 || null,
      buildings: Array.isArray(result?.buildings) ? result.buildings : [],
      selectionError: result?.selection_error || null
    });
  }, []);

  const handleConfirmSelection = useCallback(() => {
    // Lock the current map-based selection as the confirmed one.
    if (!selection.count || !selection.selectedGeoJSON) return;
    setConfirmedSelection(selection);
    setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    setBuildingAssignments({});
  }, [selection]);

  const handleResetSelection = useCallback(() => {
    // Return to drawing mode and clear all selection state.
    setConfirmedSelection(null);
    setSelection(INITIAL_SELECTION);
    setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    setBuildingAssignments({});
  }, []);

  const handleConstructionAreaSelection = useCallback((result) => {
    setConstructionAreaSelection({
      count: result?.count || 0,
      selectedGeoJSON: parseSelectedGeoJSON(result?.selected_geojson),
      buildings: Array.isArray(result?.buildings) ? result.buildings : [],
      buildingKeys: Array.isArray(result?.building_keys) ? result.building_keys : [],
      selectionError: result?.selection_error || null
    });
  }, []);

  const handleConfirmConstructionFeatures = useCallback(
    ({ useTypeSelections }) => {
      if (!Array.isArray(constructionAreaSelection.buildingKeys) || !constructionAreaSelection.buildingKeys.length) {
        return;
      }

      const nextAssignments = {};
      constructionAreaSelection.buildingKeys.forEach((key, index) => {
        const building = constructionAreaSelection.buildings[index] || {};
        const useType = String(building.cea_use_type1 || "").toUpperCase();
        const mapboxType = normalizeMapboxType(
          building.mapbox_type || building.type || building.class || building.building
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
          selected.refurbishment_type,
          selected.detail,
          selected.year_start,
          selected.year_end
        );
        if (!row) return;

        nextAssignments[key] = {
          const_type: row.const_type,
          year_start: selected.year_start,
          year_end: selected.year_end,
          refurbishment_type: selected.refurbishment_type,
          detail: selected.detail,
          mapbox_type: mapboxType,
          cea_use_type1: useType
        };
      });

      if (!Object.keys(nextAssignments).length) return;

      setBuildingAssignments((prev) => ({ ...prev, ...nextAssignments }));
      setConstructionAreaSelection(INITIAL_CONSTRUCTION_AREA_SELECTION);
    },
    [constructionAreaSelection, constructionMappingRows]
  );

  const runSimulation = useCallback(() => {
    // Run against confirmed selection when present, otherwise current draft selection.
    const targetSelection = confirmedSelection || selection;
    if (!targetSelection.selectedGeoJSON || targetSelection.count <= 0) return;
    // Placeholder: simulation endpoint / workflow can be wired here.
    // For now, just confirm the action happened.
    alert(`Run simulation: ${targetSelection.count} selected building(s)`);
  }, [confirmedSelection, selection]);

  const handleSendChat = useCallback(async () => {
    const message = chatInput.trim();
    if (!message || chatLoading) return;

    setChatError(null);
    setChatMessages((prev) => [...prev, { role: "user", text: message }]);
    setChatInput("");
    setChatLoading(true);
    try {
      const reply = await sendChatMessage(message);
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: reply || "No response." }
      ]);
    } catch (e) {
      setChatError(e?.message || "Chat request failed");
    } finally {
      setChatLoading(false);
    }
  }, [chatInput, chatLoading]);

  const handleSaveScenario = useCallback(() => {
    const name = scenarioName.trim();
    if (!name) return;
    setSavedScenarios((prev) =>
      prev.includes(name) ? prev : [name, ...prev].slice(0, 8)
    );
    setScenarioName("");
  }, [scenarioName]);

  useEffect(() => {
    // In SHP mode, preload all buildings from the backend once.
    if (USE_MAPBOX_BUILDINGS) return;
    let cancelled = false;
    async function load() {
      try {
        const geojson = await fetchBuildings();
        if (!cancelled) setBuildingsGeoJSON(geojson);
      } catch (e) {
        if (!cancelled)
          setLoadError(e?.message || "Failed to load buildings GeoJSON");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    // Load construction decomposition mapping once for construction-phase options.
    let cancelled = false;
    async function load() {
      try {
        const rows = await fetchConstructionTypeMapping();
        if (!cancelled) {
          const normalizedRows = Array.isArray(rows) ? rows : [];
          setConstructionMappingRows(normalizedRows);
          setConstructionMappingError(
            normalizedRows.length
              ? null
              : "Construction mapping is empty. Check CSV headers/content."
          );
        }
      } catch (e) {
        if (!cancelled) {
          setConstructionMappingRows([]);
          setConstructionMappingError(
            e?.message || "Failed to load construction type mapping"
          );
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const confirmedFeatureCollection = parseSelectedGeoJSON(
    confirmedSelection?.selectedGeoJSON
  );

  const lockedSelectionGeoJSONWithState = (() => {
    if (!confirmedFeatureCollection?.features?.length) return null;
    const total = confirmedFeatureCollection.features.length;
    const assignedCount = confirmedFeatureCollection.features.reduce((acc, feature, index) => {
      const key = getFeatureStableKey(feature, index);
      return acc + (buildingAssignments[key] ? 1 : 0);
    }, 0);
    const allAssigned = total > 0 && assignedCount === total;

    return {
      type: "FeatureCollection",
      features: confirmedFeatureCollection.features.map((feature, index) => {
        const key = getFeatureStableKey(feature, index);
        const assignment = buildingAssignments[key] || null;
        const assignmentState = allAssigned ? "complete" : assignment ? "defined" : "pending";
        return {
          ...feature,
          properties: {
            ...(feature?.properties || {}),
            __selection_key: key,
            __assignment_state: assignmentState,
            const_type: assignment?.const_type || null,
            refurbishment_type: assignment?.refurbishment_type || null,
            detail: assignment?.detail || null,
            feature_year_start: assignment?.year_start || null,
            feature_year_end: assignment?.year_end || null
          }
        };
      })
    };
  })();

  const confirmedBuildingsWithAssignments =
    lockedSelectionGeoJSONWithState?.features?.map((feature) => feature.properties) || [];

  const definedBuildingCount = confirmedBuildingsWithAssignments.filter(
    (props) => props?.const_type
  ).length;
  const allConstructionDefined =
    confirmedBuildingsWithAssignments.length > 0 &&
    definedBuildingCount === confirmedBuildingsWithAssignments.length;

  const activeSelection = confirmedSelection
    ? {
        ...confirmedSelection,
        count: confirmedBuildingsWithAssignments.length,
        buildings: confirmedBuildingsWithAssignments,
        selectedGeoJSON: lockedSelectionGeoJSONWithState
      }
    : selection;
  const hasSelection = activeSelection.count > 0 && Boolean(activeSelection.selectedGeoJSON);

  return (
    <div
      className={[
        "app-root",
        sidebarHidden ? "sidebar-hidden" : "",
        leftCollapsed ? "left-collapsed" : "",
        rightCollapsed ? "right-collapsed" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <LeftDock
        sidebarHidden={sidebarHidden}
        scenarioName={scenarioName}
        setScenarioName={setScenarioName}
        handleSaveScenario={handleSaveScenario}
        savedScenarios={savedScenarios}
        hasSelection={hasSelection}
        runSimulation={runSimulation}
        setSidebarHidden={setSidebarHidden}
      />

      <div className="center-column">
        <button
          type="button"
          className="sidebar-toggle-btn"
          aria-label={sidebarHidden ? "Show sidebar" : "Hide sidebar"}
          onClick={() => setSidebarHidden(!sidebarHidden)}
        >
          {sidebarHidden ? "▶" : "◀"}
        </button>
        <MapView
          buildingsGeoJSON={buildingsGeoJSON}
          selectedGeoJSON={selection.selectedGeoJSON}
          lockedSelectionGeoJSON={lockedSelectionGeoJSONWithState}
          selectionLocked={Boolean(confirmedSelection)}
          constructionPhaseActive={Boolean(confirmedSelection)}
          onSelection={handleSelection}
          onConstructionAreaSelection={handleConstructionAreaSelection}
        />
        {loadError && (
          <div
            className="map-overlay map-overlay-error"
            style={{
              position: "absolute",
              left: 12,
              bottom: 12,
              padding: "8px 10px",
              borderRadius: 8,
              background: "rgba(0,0,0,0.75)",
              color: "white",
              fontSize: 12,
              maxWidth: 420
            }}
          >
            Buildings load error: {loadError}
          </div>
        )}
        <ChatPanel
          leftCollapsed={leftCollapsed}
          setLeftCollapsed={setLeftCollapsed}
          activeSelectionCount={activeSelection.count}
          chatMessages={chatMessages}
          chatLoading={chatLoading}
          chatError={chatError}
          chatInput={chatInput}
          setChatInput={setChatInput}
          handleSendChat={handleSendChat}
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
      </div>
    </div>
  );
}

export default App;

