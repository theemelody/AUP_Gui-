import { useCallback, useState } from 'react';

export interface TextSelectionState {
  text: string;
  anchorX: number;
  anchorY: number;
}

/**
 * Tracks the browser's current text selection within a container element.
 * Returns the selected text and the position of the selection anchor for
 * positioning the comment popover.
 */
export function useTextSelection() {
  const [selection, setSelection] = useState<TextSelectionState | null>(null);

  const onSelectionChange = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.toString().trim()) {
      setSelection(null);
      return;
    }
    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    setSelection({
      text: sel.toString(),
      anchorX: rect.left + rect.width / 2,
      anchorY: rect.top,
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelection(null);
    window.getSelection()?.removeAllRanges();
  }, []);

  return { selection, onSelectionChange, clearSelection };
}
