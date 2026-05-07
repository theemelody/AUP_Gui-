# Styles

Token-driven CSS design system. All design tokens live in `_variables.css`. Each file covers one feature area and imports nothing from the others — only tokens from `_variables.css`.

## File structure

| File | Purpose | Lines |
| --- | --- | --- |
| `_variables.css` | All design tokens — colors, spacing, radii, blur, transitions, button variants | 118 |
| `base.css` | HTML/body reset, global form element defaults | 33 |
| `layout.css` | App grid, map container, sidebar toggle | 41 |
| `left-dock.css` | Left sidebar: header, tabs, scenario chips, simulation log | 251 |
| `chat-panel.css` | Floating chat window, messages, input bar | 103 |
| `bottom-panels.css` | Floating panel container shell (selection + construction) | 103 |
| `selection-panel.css` | Selected buildings list and building cards | 103 |
| `construction-panel.css` | Construction phase UI, cascade dropdowns | 173 |
| `common.css` | Shared button/input base, status indicators, panel utilities | 222 |
| `techtree.css` | Tech Tree ReactFlow canvas, layer bands, node pills | 447 |

All files are imported in order in `../index.css`.

## Two visual zones

The app has two distinct visual contexts. Styles must stay within their zone.

| Zone | Used in | Character |
| --- | --- | --- |
| **Dark** | Left dock, tech tree background | Navy/slate bg, light text, sky-blue accent |
| **Light** | Map-overlay panels (chat, right, selection) | Frosted glass, white cards, dark text |

Buttons and inputs follow the zone they live in. Shared button tokens in `_variables.css` encode both zones (`--btn-dark-*`, `--btn-light-*`) so mixing is explicit, not accidental.

## Design tokens (`_variables.css`)

Token groups in `:root`:

- **Colors** — `--color-dark-bg`, `--color-dark-text`, `--color-accent-sky`, `--color-light-bg`, `--color-light-text`, `--color-light-border`, status variants (`--color-status-complete`, `--color-status-running`, etc.)
- **Spacing** — `--spacing-xs` (4px) through `--spacing-2xl` (16px)
- **Typography** — `--font-size-xs` through `--font-size-lg`, `--font-stack`
- **Geometry** — `--radius-sm` / `--radius-md` / `--radius-lg`, `--shadow-*`, `--blur-md`
- **Transitions** — `--transition-quick` (120ms), `--transition-fast` (180ms)
- **Button variants** — `--btn-dark-bg/hover/color`, `--btn-light-bg/hover/color`, `--btn-primary-*`, `--btn-success-*`, `--btn-danger-*`
- **Panel backgrounds** — `--panel-bg`, `--panel-bg-subtle`, `--card-light-bg`, `--color-dark-bg-55`, `--color-dark-bg-95`

## Naming conventions

- **BEM-like**: `.component-section`, `.component-section-title`
- **State modifiers**: `.is-active`, `.is-hidden`, `.is-collapsed`, `.is-disabled`
- **Semantic roles**: `.action-button`, `.status-dot`, `.construction-error`
- **Namespace prefixes**: `.left-dock-`, `.construction-`, `.chat-`, `.selected-`, `.bottom-panel-`
- **Utility classes** (in `common.css`): `.panel-blur`, `.panel-header`, `.btn-dark`, `.btn-light`, `.btn-primary`, `.btn-success`, `.btn-danger`

## Adding a new component

1. Create `styles/my-feature.css` using tokens only — no hardcoded hex or px values.
2. Import it at the bottom of `../index.css`.

```css
/* styles/my-feature.css */
.my-feature {
  padding: var(--spacing-lg);
  border-radius: var(--radius-md);
  background: var(--panel-bg);
  -webkit-backdrop-filter: var(--blur-md);
  backdrop-filter: var(--blur-md);
  color: var(--color-light-text);
  transition: background var(--transition-fast);
}
```

For dark-zone components use `--color-dark-bg-55` / `--color-dark-text` instead. Never mix zone tokens in the same component.
