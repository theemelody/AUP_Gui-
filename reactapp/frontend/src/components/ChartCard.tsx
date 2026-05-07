interface ChartCardProps {
  title: string;
  subtitle?: string;
  height?: number;
  locked?: boolean;
  lockMessage?: string;
  wide?: boolean;
  children?: React.ReactNode;
}

function LockIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

export default function ChartCard({
  title,
  subtitle,
  height = 340,
  locked = false,
  lockMessage = 'Run simulation to unlock',
  wide = false,
  children,
}: ChartCardProps) {
  return (
    <div className={`chart-card${wide ? ' chart-card--wide' : ''}`}>
      <div className="chart-card__header">
        <span className="chart-card__title">{title}</span>
        {subtitle && <span className="chart-card__subtitle">{subtitle}</span>}
      </div>
      {locked ? (
        <div className="chart-card__locked" style={{ height }}>
          <LockIcon />
          <span>{lockMessage}</span>
        </div>
      ) : (
        <div style={{ height }}>{children}</div>
      )}
    </div>
  );
}
