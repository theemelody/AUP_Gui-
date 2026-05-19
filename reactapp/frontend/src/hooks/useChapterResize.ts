import { useCallback, useRef } from 'react';

export type ResizeAxis = 'horizontal' | 'vertical';

/**
 * Returns a mousedown handler for drag-resize separators.
 * Mutates CSS custom properties on the provided container ref.
 *
 * Usage:
 *   const { onSepMouseDown } = useChapterResize(containerRef);
 *   <div onMouseDown={(e) => onSepMouseDown(e, 'horizontal', '--secap-row1-height')} />
 */
export function useChapterResize(containerRef: React.RefObject<HTMLElement | null>) {
  const dragRef = useRef<{
    axis: ResizeAxis;
    prop: string;
    startPx: number;
    startValue: number;
  } | null>(null);

  const onSepMouseDown = useCallback(
    (e: React.MouseEvent, axis: ResizeAxis, cssProp: string) => {
      e.preventDefault();
      const el = containerRef.current;
      if (!el) return;
      const startValue = parseFloat(
        getComputedStyle(el).getPropertyValue(cssProp) || '300'
      );
      dragRef.current = {
        axis,
        prop: cssProp,
        startPx: axis === 'vertical' ? e.clientX : e.clientY,
        startValue,
      };

      const onMove = (ev: MouseEvent) => {
        const d = dragRef.current;
        if (!d || !containerRef.current) return;
        const delta =
          d.axis === 'vertical' ? ev.clientX - d.startPx : ev.clientY - d.startPx;
        const next = Math.max(100, d.startValue + delta);
        containerRef.current.style.setProperty(d.prop, `${next}px`);
      };

      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      };

      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    },
    [containerRef],
  );

  return { onSepMouseDown };
}
