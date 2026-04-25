# React Feature Components Overview

App is now split into focused feature components to make development easier for new React contributors.

## LeftDock

LeftDock renders the full left sidebar and keeps UI-only behavior related to workspace controls.

It includes:

- model display and status rows
- scenario input and save action
- saved scenarios list
- run simulation button

App still owns the state and handlers, while LeftDock receives props for values and callbacks.

## ChatPanel

ChatPanel renders the floating chat panel over the map.

It is responsible for:

- collapse and expand interaction for the left bottom panel
- displaying the message stream
- showing loading and error states
- handling input and send action from props

The chat network request logic remains in App.

## SelectionPanel

SelectionPanel renders the bottom-center selected buildings list and selection actions.

It handles:

- draft vs confirmed title state
- confirm/reset action button
- selection error display
- building card rendering and key ordering

This component centralizes all selection-list UI so App does not contain long card-mapping JSX.

## RightPanel

RightPanel renders the active construction-type phase tools.

It currently controls:

- collapse and expand behavior
- construction-phase status summary (confirmed / defined counts)
- per-use-type and per-mapbox-type filtered selectors
- refurbishment, detail, and year-range selection
- confirm-features action button
- mapping and selection error display

It receives normalized mapping rows and current construction-area selection from App.

## MapView

MapView remains the map interaction boundary.

It owns:

- map creation and draw controls
- frontend spatial intersection selection
- mapbox type to CEA use-type enrichment
- phase-aware draw behavior:
  - initial building selection
  - construction-area selection over confirmed buildings
- confirmed building rendering with state colors:
  - pending = orange
  - defined = yellow
  - complete = green
- map resize handling for stable draw/pointer alignment

## App

App is now the composition and state orchestration layer.

It owns:

- global UI state and workflow state
- side effects (data load, chat request)
- construction mapping fetch and normalization consumption
- construction assignment state keyed per confirmed building
- derivation of locked/confirmed GeoJSON enriched with assignment state
- lazy-loading of `MapView` to split heavy map dependencies from the initial app bundle
- passing data and callbacks into child feature components

This separation follows a common industry pattern: App as container, feature components as presentational and interaction boundaries.

## Shared UI Components

Reusable UI primitives now live in `components/common/`:

- `CollapsiblePanel.jsx`
  - shared shell used by `ChatPanel` and `RightPanel`
- `LabeledSelectField.jsx`
  - shared labeled select used by RightPanel configuration controls
- `LeftDockTab.jsx`
  - shared tab wrapper used by LeftDock accordion sections

## Shared Selection Utilities

Selection and mapbox normalization helpers are centralized in `src/utils/selection.js`.

This module includes:

- GeoJSON parsing helper
- stable feature key helper
- mapbox type normalization helpers for single values and lists
- building mapbox type extraction helper
