function CollapsiblePanel({
  positionClass,
  collapsed,
  setCollapsed,
  title,
  titleSuffix = "",
  expandAriaLabel,
  collapseAriaLabel,
  children
}) {
  return (
    <aside
      className={[
        "bottom-panel",
        positionClass,
        collapsed ? "is-collapsed" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="bottom-panel-header">
        <div className="bottom-panel-title">
          {title}
          {titleSuffix ? ` - ${titleSuffix}` : ""}
        </div>
        <button
          type="button"
          className="bottom-panel-toggle"
          aria-label={collapsed ? expandAriaLabel : collapseAriaLabel}
          onClick={() => setCollapsed((v) => !v)}
        >
          {collapsed ? "▲" : "▼"}
        </button>
      </div>
      <div className="bottom-panel-body">{children}</div>
    </aside>
  );
}

export default CollapsiblePanel;
