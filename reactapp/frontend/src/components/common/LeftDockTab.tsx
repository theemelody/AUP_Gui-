interface LeftDockTabProps {
  id: string;
  title: string;
  active: boolean;
  onActivate: () => void;
  children?: React.ReactNode;
}

function LeftDockTab({ id, title, active, onActivate, children }: LeftDockTabProps) {
  return (
    <section
      className={['left-dock-tab', active ? 'is-active' : ''].filter(Boolean).join(' ')}
    >
      <button
        type="button"
        className="left-dock-tab-header"
        aria-expanded={active}
        aria-controls={`tab-panel-${id}`}
        onClick={onActivate}
      >
        {title}
      </button>
      <div id={`tab-panel-${id}`} className="left-dock-tab-panel">
        {children}
      </div>
    </section>
  );
}

export default LeftDockTab;
