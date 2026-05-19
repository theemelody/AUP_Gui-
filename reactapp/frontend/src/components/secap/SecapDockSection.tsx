import { useSecapContext, type ChapterId } from '../../context/SecapContext.js';
import SecapStatusBadge from './SecapStatusBadge.js';
import SecapContextMeter from './SecapContextMeter.js';

const CHAPTER_LABELS: Record<ChapterId, string> = {
  1: 'Ch.1 Executive Summary',
  2: 'Ch.2 Strategy',
  3: 'Ch.3 BEI',
  4: 'Ch.4 RVA',
  5: 'Ch.5 Mitigation',
  6: 'Ch.6 Adaptation',
};

const CHAPTER_IDS: ChapterId[] = [1, 2, 3, 4, 5, 6];

export default function SecapDockSection() {
  const {
    chapters,
    orchestratorStatus,
    orchestratorNotifications,
    availableModels,
    setChapterModel,
    generateChapter,
    stopChapter,
    saveChapter,
    dismissNotification,
  } = useSecapContext();

  const scrollToChapter = (id: ChapterId) => {
    document.getElementById(`secap-chapter-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="secap-dock">
      <div className="secap-dock__section-label">CHAPTERS</div>
      <div className="secap-dock__chapters">
        {CHAPTER_IDS.map((id) => {
          const ch = chapters[id];
          return (
            <div key={id} className="secap-dock__chapter-row" onClick={() => scrollToChapter(id)}>
              <div className="secap-dock__chapter-top">
                <span className="secap-dock__chapter-label">{CHAPTER_LABELS[id]}</span>
                <SecapStatusBadge status={ch.status} />
              </div>

              <div className="secap-dock__chapter-controls" onClick={(e) => e.stopPropagation()}>
                <select
                  className="secap-dock__model-select"
                  value={ch.model}
                  onChange={(e) => setChapterModel(id, e.target.value)}
                  title="Select LLM model for this chapter"
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>

                {ch.status === 'working' || ch.status === 'reading' ? (
                  <button
                    className="secap-dock__btn secap-dock__btn--stop"
                    onClick={() => stopChapter(id)}
                    title="Stop generation"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    className="secap-dock__btn secap-dock__btn--generate"
                    onClick={() => generateChapter(id)}
                    title="Generate with LLM"
                  >
                    Generate
                  </button>
                )}

                {ch.isDirty && (
                  <button
                    className="secap-dock__btn secap-dock__btn--save"
                    onClick={() => saveChapter(id)}
                    title="Save chapter"
                  >
                    Save
                  </button>
                )}
              </div>

              <SecapContextMeter used={ch.contextTokensUsed} max={ch.contextTokensMax} />

              {ch.error && (
                <div className="secap-dock__chapter-error" title={ch.error}>
                  {ch.error.slice(0, 60)}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="secap-dock__section-label">ORCHESTRATOR</div>
      <div className={`secap-dock__orch-status secap-dock__orch-status--${orchestratorStatus}`}>
        {orchestratorStatus === 'working' ? 'Working…' : 'Idle'}
      </div>

      {orchestratorNotifications.length > 0 && (
        <div className="secap-dock__notifications">
          {orchestratorNotifications.map((n, i) => (
            <div key={i} className="secap-dock__notification">
              <span className="secap-dock__notif-text">
                {n.triggerType === 'chapter_saved'
                  ? `Ch.${n.sourceId} saved → Ch.${n.affected.join(', ')} affected`
                  : `Data source '${n.sourceId}' → Ch.${n.affected.join(', ')} affected`}
                <br />
                <em>{n.reason}</em>
              </span>
              <button
                className="secap-dock__notif-dismiss"
                onClick={() => dismissNotification(i)}
                title="Dismiss"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
