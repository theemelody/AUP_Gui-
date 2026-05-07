import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  fetchScenarios,
  fetchScenarioData,
  fetchScenarioStatus,
  saveScenarioBuildings,
} from '../services/api.js';
import type { ActiveSelectionInfo } from '../pages/buildingSelection';
import type { PageId } from '../states/navigationMachine';

type StepStatus = 'idle' | 'running' | 'done' | 'error';
type LogEntry = { step: string; status: StepStatus; messages: string[] };
type SimStatus = 'idle' | 'running' | 'done' | 'failed';
export type ScenarioStatus = 'complete' | 'ready' | 'incomplete' | 'missing';

export interface LoadedScenario {
  geojson: unknown;
  drawnPolygon: unknown;
}

export interface ScenarioContextValue {
  scenarioName: string;
  setScenarioName: (name: string) => void;
  confirmedScenarioName: string;
  savedScenarios: string[];
  selectedScenarioForSim: string;
  scenarioStatuses: Record<string, ScenarioStatus>;
  simulationLog: LogEntry[];
  simulationStatus: SimStatus;
  loadedScenario: LoadedScenario | null;
  // Actions
  saveScenario: () => Promise<void>;
  selectScenarioForSim: (name: string) => Promise<void>;
  runSimulation: () => void;
}

const ScenarioContext = createContext<ScenarioContextValue | null>(null);

interface ScenarioProviderProps {
  children: ReactNode;
  navigate: (page: PageId) => void;
  /** Callback returning the current active selection (managed by BuildingSelection). */
  getActiveSelection: () => ActiveSelectionInfo | null;
  /** Callback returning the current drawn polygon (managed by BuildingSelection). */
  getDrawnPolygon: () => unknown;
}

export function ScenarioProvider({
  children,
  navigate,
  getActiveSelection,
  getDrawnPolygon,
}: ScenarioProviderProps) {
  const [scenarioName, setScenarioName] = useState('');
  const [confirmedScenarioName, setConfirmedScenarioName] = useState('');
  const [savedScenarios, setSavedScenarios] = useState<string[]>([]);
  const [selectedScenarioForSim, setSelectedScenarioForSim] = useState('');
  const [scenarioStatuses, setScenarioStatuses] = useState<Record<string, ScenarioStatus>>({});
  const [simulationLog, setSimulationLog] = useState<LogEntry[]>([]);
  const [simulationStatus, setSimulationStatus] = useState<SimStatus>('idle');
  const [loadedScenario, setLoadedScenario] = useState<LoadedScenario | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // Load saved scenarios on mount
  useEffect(() => {
    fetchScenarios()
      .then((names) => setSavedScenarios(names))
      .catch(() => {/* silently ignore if backend is not running */});
  }, []);

  // Refresh scenario statuses whenever the list changes
  useEffect(() => {
    if (!savedScenarios.length) return;
    savedScenarios.forEach((name) => {
      fetchScenarioStatus(name)
        .then((status) =>
          setScenarioStatuses((prev) => ({ ...prev, [name]: status as ScenarioStatus })),
        )
        .catch(() => {/* ignore */});
    });
  }, [savedScenarios]);

  const saveScenario = useCallback(async () => {
    const name = scenarioName.trim();
    if (!name) return;
    const sel = getActiveSelection();
    if (!sel?.selectedGeoJSON || !sel?.count) {
      alert('Please select buildings first before saving a scenario.');
      return;
    }
    try {
      const result = await saveScenarioBuildings(
        sel.selectedGeoJSON,
        name,
        getDrawnPolygon(),
      );
      if (result.success) {
        setSavedScenarios((prev: string[]) =>
          prev.includes(name) ? prev : [name, ...prev].slice(0, 8),
        );
        setConfirmedScenarioName(name);
        fetchScenarioStatus(name).then((status) =>
          setScenarioStatuses((prev) => ({ ...prev, [name]: status as ScenarioStatus })),
        );
        alert(`Scenario '${name}' saved successfully to ${result.scenario_path}`);
      }
    } catch (e) {
      alert((e as Error)?.message || 'Failed to save scenario');
    }
  }, [scenarioName, getActiveSelection, getDrawnPolygon]);

  const selectScenarioForSim = useCallback(async (name: string) => {
    setSelectedScenarioForSim(name);
    if (!name) { setLoadedScenario(null); return; }
    try {
      const data = await fetchScenarioData(name);
      setLoadedScenario({ geojson: data.selected_geojson, drawnPolygon: data.drawn_polygon });
      setConfirmedScenarioName(name);
      navigate('simulation' as PageId);
    } catch (e) {
      console.warn('[ScenarioContext] scenario.json not found, skipping load:', e);
    }
  }, [navigate]);

  const runSimulation = useCallback(() => {
    const name = selectedScenarioForSim;
    if (!name) return;

    esRef.current?.close();
    setSimulationLog([]);
    setSimulationStatus('running');

    const es = new EventSource(`/api/run-simulation?scenario_name=${encodeURIComponent(name)}`);
    esRef.current = es;

    es.onmessage = (evt) => {
      const data = JSON.parse(evt.data);

      if (data.status === 'complete' || data.status === 'failed') {
        setSimulationStatus(data.status === 'complete' ? 'done' : 'failed');
        if (data.status === 'complete') {
          setScenarioStatuses((prev) => ({ ...prev, [name]: 'complete' }));
        }
        es.close();
        return;
      }

      setSimulationLog((prev) => {
        const idx = prev.findIndex((e) => e.step === data.step);
        const entry: LogEntry =
          idx >= 0
            ? {
                ...prev[idx],
                status: data.status,
                messages: data.message
                  ? [...prev[idx].messages, data.message]
                  : prev[idx].messages,
              }
            : {
                step: data.step,
                status: data.status,
                messages: data.message ? [data.message] : [],
              };
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = entry;
          return next;
        }
        return [...prev, entry];
      });
    };

    es.onerror = () => {
      setSimulationStatus('failed');
      es.close();
    };
  }, [selectedScenarioForSim]);

  const value = useMemo<ScenarioContextValue>(
    () => ({
      scenarioName,
      setScenarioName,
      confirmedScenarioName,
      savedScenarios,
      selectedScenarioForSim,
      scenarioStatuses,
      simulationLog,
      simulationStatus,
      loadedScenario,
      saveScenario,
      selectScenarioForSim,
      runSimulation,
    }),
    [
      scenarioName, confirmedScenarioName, savedScenarios, selectedScenarioForSim,
      scenarioStatuses, simulationLog, simulationStatus, loadedScenario,
      saveScenario, selectScenarioForSim, runSimulation,
    ],
  );

  return (
    <ScenarioContext.Provider value={value}>
      {children}
    </ScenarioContext.Provider>
  );
}

export function useScenarioContext(): ScenarioContextValue {
  const ctx = useContext(ScenarioContext);
  if (!ctx) throw new Error('useScenarioContext must be used inside ScenarioProvider');
  return ctx;
}
