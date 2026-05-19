import { useRef } from 'react';
import { useChapterResize } from '../../hooks/useChapterResize.js';
import { useSecapContext } from '../../context/SecapContext.js';
import SecapChapterPanel from './SecapChapterPanel.js';
import SecapFullTextPanel from './SecapFullTextPanel.js';
import type { ChapterId } from '../../context/SecapContext.js';

export default function SecapLayout() {
  const { isLoading } = useSecapContext();
  const containerRef = useRef<HTMLDivElement>(null);
  const { onSepMouseDown } = useChapterResize(containerRef);

  if (isLoading) {
    return (
      <div className="secap-layout secap-layout--loading">
        <div className="secap-layout__loading-msg">Loading chapters…</div>
      </div>
    );
  }

  return (
    <div className="secap-layout" ref={containerRef}>
      {/* Row 1: Chapters 1–3 */}
      <div className="secap-layout__row secap-layout__row--top">
        <SecapChapterPanel chapterId={1 as ChapterId} />
        <div
          className="secap-layout__vsep"
          onMouseDown={(e) => onSepMouseDown(e, 'vertical', '--secap-col1-width')}
        />
        <SecapChapterPanel chapterId={2 as ChapterId} />
        <div
          className="secap-layout__vsep"
          onMouseDown={(e) => onSepMouseDown(e, 'vertical', '--secap-col2-width')}
        />
        <SecapChapterPanel chapterId={3 as ChapterId} />
      </div>

      {/* Horizontal separator between rows */}
      <div
        className="secap-layout__hsep"
        onMouseDown={(e) => onSepMouseDown(e, 'horizontal', '--secap-row1-height')}
      />

      {/* Row 2: Chapters 4–6 */}
      <div className="secap-layout__row secap-layout__row--bottom">
        <SecapChapterPanel chapterId={4 as ChapterId} />
        <div
          className="secap-layout__vsep"
          onMouseDown={(e) => onSepMouseDown(e, 'vertical', '--secap-col1-width')}
        />
        <SecapChapterPanel chapterId={5 as ChapterId} />
        <div
          className="secap-layout__vsep"
          onMouseDown={(e) => onSepMouseDown(e, 'vertical', '--secap-col2-width')}
        />
        <SecapChapterPanel chapterId={6 as ChapterId} />
      </div>

      {/* Horizontal separator before full text */}
      <div
        className="secap-layout__hsep"
        onMouseDown={(e) => onSepMouseDown(e, 'horizontal', '--secap-row2-height')}
      />

      {/* Row 3: Full text panel (spans all 3 columns) */}
      <div className="secap-layout__row secap-layout__row--full">
        <SecapFullTextPanel />
      </div>
    </div>
  );
}
