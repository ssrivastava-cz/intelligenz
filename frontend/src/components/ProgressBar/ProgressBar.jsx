import "./ProgressBar.css";

/**
 * Animated horizontal progress bar.
 * @param {{value: number}} props - percentage from 0 to 100
 */
function ProgressBar({ value }) {
  const clamped = Math.min(100, Math.max(0, value));

  return (
    <div className="progress-bar" role="progressbar" aria-valuenow={clamped} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-bar__fill" style={{ width: `${clamped}%` }} />
    </div>
  );
}

export default ProgressBar;
