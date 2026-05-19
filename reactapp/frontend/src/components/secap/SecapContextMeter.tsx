interface SecapContextMeterProps {
  used: number;
  max: number;
}

export default function SecapContextMeter({ used, max }: SecapContextMeterProps) {
  const pct = max > 0 ? Math.min(100, Math.round((used / max) * 100)) : 0;
  const fill = pct > 85 ? 'secap-meter__fill--high' : pct > 60 ? 'secap-meter__fill--mid' : '';
  return (
    <div className="secap-meter" title={`${used} / ${max} tokens (${pct}%)`}>
      <div className={`secap-meter__fill ${fill}`} style={{ width: `${pct}%` }} />
      <span className="secap-meter__label">{pct}%</span>
    </div>
  );
}
