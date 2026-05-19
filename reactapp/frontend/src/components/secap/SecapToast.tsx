import { useEffect } from 'react';

interface SecapToastProps {
  message: string;
  onDismiss: () => void;
  durationMs?: number;
}

export default function SecapToast({ message, onDismiss, durationMs = 2500 }: SecapToastProps) {
  useEffect(() => {
    const t = setTimeout(onDismiss, durationMs);
    return () => clearTimeout(t);
  }, [onDismiss, durationMs]);

  return (
    <div className="secap-toast" onClick={onDismiss}>
      <span className="secap-toast__icon">🔒</span>
      {message}
    </div>
  );
}
