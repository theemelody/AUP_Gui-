import { useState } from 'react';
import type { TextSelectionState } from '../../hooks/useTextSelection.js';

interface SecapCommentPopoverProps {
  selection: TextSelectionState;
  onSubmit: (comment: string) => void;
  onClose: () => void;
}

export default function SecapCommentPopover({ selection, onSubmit, onClose }: SecapCommentPopoverProps) {
  const [comment, setComment] = useState('');

  const handleSubmit = () => {
    if (!comment.trim()) return;
    onSubmit(comment.trim());
    setComment('');
    onClose();
  };

  return (
    <div
      className="secap-comment-popover"
      style={{
        position: 'fixed',
        left: selection.anchorX,
        top: selection.anchorY - 8,
        transform: 'translate(-50%, -100%)',
        zIndex: 1000,
      }}
    >
      <div className="secap-comment-popover__selection">
        &ldquo;{selection.text.slice(0, 80)}{selection.text.length > 80 ? '…' : ''}&rdquo;
      </div>
      <textarea
        className="secap-comment-popover__input"
        placeholder="Add a comment for the agent…"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        rows={3}
        autoFocus
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit();
          if (e.key === 'Escape') onClose();
        }}
      />
      <div className="secap-comment-popover__actions">
        <button className="secap-comment-popover__submit" onClick={handleSubmit}>
          Request revision
        </button>
        <button className="secap-comment-popover__cancel" onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
}
