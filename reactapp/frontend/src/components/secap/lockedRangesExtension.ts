import { Decoration, type DecorationSet, EditorView, ViewPlugin, type ViewUpdate } from '@codemirror/view';
import { StateField, StateEffect, RangeSetBuilder, type Transaction } from '@codemirror/state';

export interface LockedLineRange {
  startLine: number; // 1-indexed
  endLine: number;   // 1-indexed, inclusive
}

/** Effect to push a new set of locked ranges into the editor state. */
export const setLockedRangesEffect = StateEffect.define<LockedLineRange[]>();

/** State field holding the current locked ranges. */
const lockedRangesField = StateField.define<LockedLineRange[]>({
  create: () => [],
  update(ranges, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setLockedRangesEffect)) return effect.value;
    }
    return ranges;
  },
});

/** Mark decoration applied to locked lines. */
const lockedLineMark = Decoration.line({ class: 'cm-locked-line' });

/** ViewPlugin that rebuilds line decorations whenever the doc or locked ranges change. */
const lockedRangesDecorations = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;

    constructor(view: EditorView) {
      this.decorations = this.build(view);
    }

    update(update: ViewUpdate) {
      if (update.docChanged || update.startState.field(lockedRangesField) !== update.state.field(lockedRangesField)) {
        this.decorations = this.build(update.view);
      }
    }

    build(view: EditorView): DecorationSet {
      const ranges = view.state.field(lockedRangesField);
      if (!ranges.length) return Decoration.none;
      const builder = new RangeSetBuilder<Decoration>();
      const doc = view.state.doc;
      for (const { startLine, endLine } of ranges) {
        const from = Math.max(1, startLine);
        const to = Math.min(doc.lines, endLine);
        for (let ln = from; ln <= to; ln++) {
          const line = doc.line(ln);
          builder.add(line.from, line.from, lockedLineMark);
        }
      }
      return builder.finish();
    }
  },
  { decorations: (v) => v.decorations },
);

/** Transaction filter: blocks changes that overlap locked ranges and calls onBlocked. */
function lockedRangesFilter(onBlocked: () => void) {
  return EditorView.updateListener.of((update) => {
    if (!update.docChanged) return;
    const tr: Transaction = update.transactions[0];
    if (!tr) return;
    const ranges = tr.startState.field(lockedRangesField);
    if (!ranges.length) return;
    const doc = tr.startState.doc;

    for (const { startLine, endLine } of ranges) {
      const from = doc.line(Math.max(1, startLine)).from;
      const to = doc.line(Math.min(doc.lines, endLine)).to;
      let blocked = false;
      tr.changes.iterChangedRanges((chFrom: number, chTo: number) => {
        if (chFrom < to && chTo > from) blocked = true;
      });
      if (blocked) {
        // Can't cancel after the fact; just notify. Phase 2 adds transactionFilter.
        onBlocked();
        break;
      }
    }
  });
}

/** Returns a CodeMirror extension bundle for locked ranges. */
export function lockedRangesExtension(onBlocked: () => void) {
  return [lockedRangesField, lockedRangesDecorations, lockedRangesFilter(onBlocked)];
}

export { lockedRangesField };
