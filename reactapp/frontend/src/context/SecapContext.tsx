import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import {
  fetchSecapChapters,
  saveSecapChapter,
  orchestrateSecap,
  openSecapGenerateStream,
  postSecapComment,
} from '../services/api.js';
import { fetchOllamaModels } from '../services/api.js';
import { useScenarioContext } from './ScenarioContext.js';

export type ChapterId = 1 | 2 | 3 | 4 | 5 | 6;
export type ChapterStatus = 'idle' | 'working' | 'reading' | 'blocked' | 'waiting';

export interface LockedRange {
  startLine: number;
  endLine: number;
}

export interface ChapterState {
  id: ChapterId;
  content: string;
  savedContent: string;
  isDirty: boolean;
  model: string;
  status: ChapterStatus;
  contextTokensUsed: number;
  contextTokensMax: number;
  lockedRanges: LockedRange[];
  error: string | null;
}

export interface OrchestratorNotification {
  triggerType: 'chapter_saved' | 'data_source_updated';
  sourceId: string;
  affected: ChapterId[];
  reason: string;
}

export interface SecapContextValue {
  chapters: Record<ChapterId, ChapterState>;
  orchestratorStatus: 'idle' | 'working';
  orchestratorNotifications: OrchestratorNotification[];
  availableModels: string[];
  isLoading: boolean;
  setChapterContent: (id: ChapterId, content: string) => void;
  setChapterModel: (id: ChapterId, model: string) => void;
  saveChapter: (id: ChapterId, contentOverride?: string) => Promise<void>;
  generateChapter: (id: ChapterId) => void;
  stopChapter: (id: ChapterId) => void;
  postComment: (id: ChapterId, selected: string, comment: string) => void;
  addLockedRange: (id: ChapterId, range: LockedRange) => void;
  removeLockedRange: (id: ChapterId, index: number) => void;
  loadAllChapters: (scenarioName: string) => Promise<void>;
  getFullText: () => string;
  dismissNotification: (index: number) => void;
}

const CHAPTER_IDS: ChapterId[] = [1, 2, 3, 4, 5, 6];

function makeInitialChapter(id: ChapterId): ChapterState {
  return {
    id,
    content: '',
    savedContent: '',
    isDirty: false,
    model: 'llama3:latest',
    status: 'idle',
    contextTokensUsed: 0,
    contextTokensMax: 4096,
    lockedRanges: [],
    error: null,
  };
}

function makeInitialChapters(): Record<ChapterId, ChapterState> {
  return Object.fromEntries(
    CHAPTER_IDS.map((id) => [id, makeInitialChapter(id)])
  ) as Record<ChapterId, ChapterState>;
}

const SecapContext = createContext<SecapContextValue | null>(null);

export function useSecapContext(): SecapContextValue {
  const ctx = useContext(SecapContext);
  if (!ctx) throw new Error('useSecapContext must be used inside SecapProvider');
  return ctx;
}

export function SecapProvider({ children }: { children: ReactNode }) {
  const { confirmedScenarioName } = useScenarioContext();

  const [chapters, setChapters] = useState<Record<ChapterId, ChapterState>>(makeInitialChapters());
  const [orchestratorStatus, setOrchestratorStatus] = useState<'idle' | 'working'>('idle');
  const [orchestratorNotifications, setOrchestratorNotifications] = useState<OrchestratorNotification[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>(['llama3:latest']);
  const [isLoading, setIsLoading] = useState(false);

  // Per-chapter SSE EventSource refs (for generate streams)
  const eventSourceRefs = useRef<Partial<Record<ChapterId, EventSource>>>({});
  // Per-chapter AbortController refs (for comment fetch streams)
  const commentAbortRefs = useRef<Partial<Record<ChapterId, AbortController>>>({});

  // Load Ollama models on mount
  useEffect(() => {
    fetchOllamaModels()
      .then((models) => { if (models?.length) setAvailableModels(models); })
      .catch(() => {});
  }, []);

  const loadAllChapters = useCallback(async (scenarioName: string) => {
    if (!scenarioName) return;
    setIsLoading(true);
    try {
      const data = await fetchSecapChapters(scenarioName);
      setChapters((prev) => {
        const next = { ...prev };
        for (const id of CHAPTER_IDS) {
          const raw = data.chapters[String(id)];
          if (raw) {
            next[id] = {
              ...prev[id],
              id,
              content: raw.content,
              savedContent: raw.content,
              isDirty: false,
              model: raw.model || prev[id].model,
              lockedRanges: raw.lockedRanges || [],
              status: 'idle',
              error: null,
            };
          }
        }
        return next;
      });
    } catch (err) {
      console.error('Failed to load SECAP chapters:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Reload chapters whenever the confirmed scenario changes
  useEffect(() => {
    if (confirmedScenarioName) {
      loadAllChapters(confirmedScenarioName);
    }
  }, [confirmedScenarioName, loadAllChapters]);

  const setChapterContent = useCallback((id: ChapterId, content: string) => {
    setChapters((prev) => ({
      ...prev,
      [id]: { ...prev[id], content, isDirty: content !== prev[id].savedContent },
    }));
  }, []);

  const setChapterModel = useCallback((id: ChapterId, model: string) => {
    setChapters((prev) => ({ ...prev, [id]: { ...prev[id], model } }));
  }, []);

  const saveChapter = useCallback(async (id: ChapterId, contentOverride?: string) => {
    if (!confirmedScenarioName) return;
    const chapter = chapters[id];
    const contentToSave = contentOverride ?? chapter.content;
    try {
      await saveSecapChapter(
        confirmedScenarioName, id, contentToSave, chapter.lockedRanges, true
      );
      setChapters((prev) => ({
        ...prev,
        [id]: { ...prev[id], content: contentToSave, savedContent: contentToSave, isDirty: false },
      }));

      // Trigger orchestration
      setOrchestratorStatus('working');
      try {
        const orch = await orchestrateSecap(confirmedScenarioName, 'chapter_saved', String(id), true);
        if (orch.affected.length > 0) {
          setOrchestratorNotifications((prev) => [
            ...prev,
            {
              triggerType: 'chapter_saved',
              sourceId: String(id),
              affected: orch.affected as ChapterId[],
              reason: orch.reason,
            },
          ]);
        }
      } finally {
        setOrchestratorStatus('idle');
      }
    } catch (err) {
      console.error('Failed to save SECAP chapter:', err);
    }
  }, [confirmedScenarioName, chapters]);

  const generateChapter = useCallback((id: ChapterId) => {
    if (!confirmedScenarioName) return;

    // Close any existing stream for this chapter
    eventSourceRefs.current[id]?.close();

    const model = chapters[id].model;
    const es = openSecapGenerateStream(confirmedScenarioName, id, model);
    eventSourceRefs.current[id] = es;

    setChapters((prev) => ({
      ...prev,
      [id]: { ...prev[id], status: 'working', error: null, content: '' },
    }));

    let accumulated = '';

    es.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data) as { token?: string; done?: boolean; error?: string };
        if (data.error) {
          setChapters((prev) => ({
            ...prev,
            [id]: { ...prev[id], status: 'idle', error: data.error ?? 'Generation failed' },
          }));
          es.close();
          return;
        }
        if (data.token) {
          accumulated += data.token;
          setChapters((prev) => ({
            ...prev,
            [id]: { ...prev[id], content: accumulated, isDirty: true },
          }));
        }
        if (data.done) {
          setChapters((prev) => ({
            ...prev,
            [id]: { ...prev[id], status: 'idle' },
          }));
          es.close();
        }
      } catch {
        // ignore parse errors mid-stream
      }
    };

    es.onerror = () => {
      setChapters((prev) => ({
        ...prev,
        [id]: { ...prev[id], status: 'idle', error: 'Generation stream error' },
      }));
      es.close();
    };
  }, [confirmedScenarioName, chapters]);

  const stopChapter = useCallback((id: ChapterId) => {
    eventSourceRefs.current[id]?.close();
    commentAbortRefs.current[id]?.abort();
    setChapters((prev) => ({ ...prev, [id]: { ...prev[id], status: 'idle' } }));
  }, []);

  const postComment = useCallback((id: ChapterId, selected: string, comment: string) => {
    if (!confirmedScenarioName) return;

    commentAbortRefs.current[id]?.abort();
    const controller = new AbortController();
    commentAbortRefs.current[id] = controller;

    setChapters((prev) => ({ ...prev, [id]: { ...prev[id], status: 'reading', error: null } }));

    postSecapComment(confirmedScenarioName, id, selected, comment)
      .then(async (resp) => {
        if (!resp.body) throw new Error('No response body');
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let revision = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() ?? '';
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6)) as { token?: string; done?: boolean; error?: string };
                if (data.token) revision += data.token;
                if (data.done) break;
              } catch { /* ignore */ }
            }
          }
        }

        // Append the revision as a suggestion below the selected text
        if (revision) {
          setChapters((prev) => ({
            ...prev,
            [id]: {
              ...prev[id],
              status: 'idle',
              content: prev[id].content + '\n\n> **Agent revision suggestion:**\n>\n> ' + revision.replace(/\n/g, '\n> '),
              isDirty: true,
            },
          }));
        } else {
          setChapters((prev) => ({ ...prev, [id]: { ...prev[id], status: 'idle' } }));
        }
      })
      .catch((err) => {
        if ((err as { name?: string }).name === 'AbortError') return;
        setChapters((prev) => ({
          ...prev,
          [id]: { ...prev[id], status: 'idle', error: String(err) },
        }));
      });
  }, [confirmedScenarioName]);

  const addLockedRange = useCallback((id: ChapterId, range: LockedRange) => {
    setChapters((prev) => ({
      ...prev,
      [id]: { ...prev[id], lockedRanges: [...prev[id].lockedRanges, range] },
    }));
  }, []);

  const removeLockedRange = useCallback((id: ChapterId, index: number) => {
    setChapters((prev) => ({
      ...prev,
      [id]: { ...prev[id], lockedRanges: prev[id].lockedRanges.filter((_, i) => i !== index) },
    }));
  }, []);

  const getFullText = useCallback(() => {
    return CHAPTER_IDS.map((id) => chapters[id].content).join('\n\n---\n\n');
  }, [chapters]);

  const dismissNotification = useCallback((index: number) => {
    setOrchestratorNotifications((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const value: SecapContextValue = {
    chapters,
    orchestratorStatus,
    orchestratorNotifications,
    availableModels,
    isLoading,
    setChapterContent,
    setChapterModel,
    saveChapter,
    generateChapter,
    stopChapter,
    postComment,
    addLockedRange,
    removeLockedRange,
    loadAllChapters,
    getFullText,
    dismissNotification,
  };

  return <SecapContext.Provider value={value}>{children}</SecapContext.Provider>;
}
