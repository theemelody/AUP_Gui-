import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useSecapContext } from '../../context/SecapContext.js';

export default function SecapFullTextPanel() {
  const { getFullText } = useSecapContext();
  const text = getFullText();

  return (
    <div className="secap-full-text-panel">
      <div className="secap-full-text-panel__header">
        <span className="secap-full-text-panel__title">Full SECAP Document</span>
        <span className="secap-full-text-panel__hint">Rendered preview — edit in panels above</span>
      </div>
      <div className="secap-full-text-panel__content secap-full-text-panel__content--markdown">
        {text.trim() ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {text}
          </ReactMarkdown>
        ) : (
          <div className="secap-full-text-panel__empty">
            Generate or edit chapters above to see the full document here.
          </div>
        )}
      </div>
    </div>
  );
}
