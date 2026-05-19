import type { ChapterStatus } from '../../context/SecapContext.js';

const STATUS_CONFIG: Record<ChapterStatus, { label: string; cssClass: string }> = {
  idle:    { label: 'Idle',    cssClass: 'secap-status--idle' },
  working: { label: 'Working', cssClass: 'secap-status--working' },
  reading: { label: 'Reading', cssClass: 'secap-status--reading' },
  blocked: { label: 'Blocked', cssClass: 'secap-status--blocked' },
  waiting: { label: 'Waiting', cssClass: 'secap-status--waiting' },
};

interface SecapStatusBadgeProps {
  status: ChapterStatus;
}

export default function SecapStatusBadge({ status }: SecapStatusBadgeProps) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.idle;
  return (
    <span className={`secap-status-badge ${cfg.cssClass}`}>
      <span className="secap-status-dot" />
      {cfg.label}
    </span>
  );
}
