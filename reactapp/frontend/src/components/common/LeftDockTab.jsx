function LeftDockTab({ id, title, active, onActivate, children }) {
  return (
    <section
      className={[
        "left-dock-tab",
        active ? "is-active" : ""
      ]
        .filter(Boolean)
        .join(" ")}
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
