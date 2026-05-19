import { useCallback, useMemo, useRef, useState } from 'react';
import CodeMirror, { type ReactCodeMirrorRef } from '@uiw/react-codemirror';
import { markdown } from '@codemirror/lang-markdown';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorView } from '@codemirror/view';
import { useSecapContext, type ChapterId, type LockedRange } from '../../context/SecapContext.js';
import { useTextSelection } from '../../hooks/useTextSelection.js';
import SecapStatusBadge from './SecapStatusBadge.js';
import SecapCommentPopover from './SecapCommentPopover.js';
import SecapToast from './SecapToast.js';
import { lockedRangesExtension, setLockedRangesEffect } from './lockedRangesExtension.js';

const CHAPTER_TITLES: Record<ChapterId, string> = {
  1: 'Executive Summary',
  2: 'Strategy',
  3: 'Baseline Emission Inventory',
  4: 'Risk & Vulnerability Assessment',
  5: 'Mitigation Actions',
  6: 'Adaptation Actions',
};

interface SecapChapterPanelProps {
  chapterId: ChapterId;
}

export default function SecapChapterPanel({ chapterId }: SecapChapterPanelProps) {
  const {
    chapters,
    setChapterContent,
    saveChapter,
    generateChapter,
    stopChapter,
    postComment,
    addLockedRange,
  } = useSecapContext();
  // setChapterContent is used only by the debounced handleChange and streaming onChange

  const chapter = chapters[chapterId];
  const { selection, onSelectionChange, clearSelection } = useTextSelection();
  const [showToast, setShowToast] = useState(false);
  const editorRef = useRef<ReactCodeMirrorRef>(null);

  const isGenerating = chapter.status === 'working' || chapter.status === 'reading';

  // Debounce context writes so 6 panels don't re-render on every keystroke.
  // CodeMirror manages its own internal doc state, so the editor stays responsive.
  // A ref holds the latest value so Save can flush it synchronously.
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null);
  const latestContentRef = useRef(chapter.content);

  const handleChange = useCallback((val: string) => {
    latestContentRef.current = val;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setChapterContent(chapterId, val), 400);
  }, [chapterId, setChapterContent]);

  const handleSave = useCallback(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    saveChapter(chapterId, latestContentRef.current);
  }, [chapterId, saveChapter]);

  const onBlocked = useCallback(() => setShowToast(true), []);

  const extensions = useMemo(
    () => [
      markdown(),
      lockedRangesExtension(onBlocked),
      EditorView.lineWrapping,
    ],
    [onBlocked],
  );

  // Sync locked ranges into CodeMirror state when they change
  const syncLockedRanges = useCallback(() => {
    const view = editorRef.current?.view;
    if (!view) return;
    view.dispatch({
      effects: setLockedRangesEffect.of(chapter.lockedRanges),
    });
  }, [chapter.lockedRanges]);

  // Lock the current selection (adds a locked range for the selected lines)
  const lockSelection = useCallback(() => {
    const view = editorRef.current?.view;
    if (!view) return;
    const { from, to } = view.state.selection.main;
    if (from === to) return; // nothing selected — don't create a range
    const doc = view.state.doc;
    const startLine = doc.lineAt(from).number;
    const endLine = doc.lineAt(to).number;
    const range: LockedRange = { startLine, endLine };
    addLockedRange(chapterId, range);
    // Immediately dispatch into the editor
    view.dispatch({ effects: setLockedRangesEffect.of([...chapter.lockedRanges, range]) });
  }, [addLockedRange, chapterId, chapter.lockedRanges]);

  return (
    <div className="secap-chapter-panel" id={`secap-chapter-${chapterId}`}>
      <div className="secap-chapter-panel__header">
        <div className="secap-chapter-panel__title">
          <span className="secap-chapter-panel__num">Ch.{chapterId}</span>
          <span className="secap-chapter-panel__name">{CHAPTER_TITLES[chapterId]}</span>
          {chapter.isDirty && <span className="secap-chapter-panel__dirty" title="Unsaved changes">●</span>}
          {chapter.lockedRanges.length > 0 && (
            <span className="secap-chapter-panel__lock-count" title={`${chapter.lockedRanges.length} locked range(s)`}>
              🔒{chapter.lockedRanges.length}
            </span>
          )}
        </div>

        <div className="secap-chapter-panel__actions">
          <SecapStatusBadge status={chapter.status} />

          <button
            className="secap-chapter-panel__btn secap-chapter-panel__btn--lock"
            onClick={lockSelection}
            title="Lock selected lines"
          >
            Lock
          </button>

          {isGenerating ? (
            <button
              className="secap-chapter-panel__btn secap-chapter-panel__btn--stop"
              onClick={() => stopChapter(chapterId)}
            >
              Stop
            </button>
          ) : (
            <button
              className="secap-chapter-panel__btn secap-chapter-panel__btn--generate"
              onClick={() => generateChapter(chapterId)}
            >
              Generate
            </button>
          )}

          <button
            className="secap-chapter-panel__btn secap-chapter-panel__btn--save"
            onClick={handleSave}
            disabled={!chapter.isDirty}
          >
            Save
          </button>
        </div>
      </div>

      {chapter.error && (
        <div className="secap-chapter-panel__error">{chapter.error}</div>
      )}

      <div
        className="secap-chapter-panel__editor-wrap"
        onMouseUp={onSelectionChange}
      >
        <CodeMirror
          ref={editorRef}
          value={chapter.content}
          onChange={isGenerating ? (val) => setChapterContent(chapterId, val) : handleChange}
          onCreateEditor={() => { setTimeout(syncLockedRanges, 50); }}
          extensions={extensions}
          theme={oneDark}
          placeholder={`Chapter ${chapterId} content — click Generate to draft with the AI agent.`}
          className="secap-chapter-panel__codemirror"
          style={{ height: '100%', fontSize: '0.78rem' }}
          basicSetup={{
            lineNumbers: false,
            foldGutter: false,
            highlightActiveLine: false,
          }}
        />
      </div>

      {selection && (
        <SecapCommentPopover
          selection={selection}
          onSubmit={(comment) => { postComment(chapterId, selection.text, comment); clearSelection(); }}
          onClose={clearSelection}
        />
      )}

      {showToast && (
        <SecapToast
          message="This section is locked and cannot be edited."
          onDismiss={() => setShowToast(false)}
        />
      )}
    </div>
  );
}
