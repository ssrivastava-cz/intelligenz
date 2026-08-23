const PATHS = {
  dashboard: (
    <>
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
    </>
  ),
  flask: <polygon points="9,3 15,3 15,8 19,20 5,20 9,8" />,
  chart: (
    <>
      <line x1="3" y1="20" x2="21" y2="20" />
      <line x1="6" y1="20" x2="6" y2="11" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="18" y1="20" x2="18" y2="15" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <polyline points="12,7 12,12 16,14" />
    </>
  ),
  upload: (
    <>
      <line x1="12" y1="3" x2="12" y2="15" />
      <polyline points="7,8 12,3 17,8" />
      <polyline points="4,15 4,19 20,19 20,15" />
    </>
  ),
  download: (
    <>
      <line x1="12" y1="3" x2="12" y2="15" />
      <polyline points="7,10 12,15 17,10" />
      <polyline points="4,15 4,19 20,19 20,15" />
    </>
  ),
  copy: (
    <>
      <rect x="4" y="8" width="12" height="12" />
      <rect x="8" y="4" width="12" height="12" />
    </>
  ),
  chevronDown: <polyline points="6,9 12,15 18,9" />,
  chevronRight: <polyline points="9,6 15,12 9,18" />,
  x: (
    <>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </>
  ),
  checkCircle: (
    <>
      <circle cx="12" cy="12" r="9" />
      <polyline points="8,12 11,15 16,9" />
    </>
  ),
  spinner: <circle cx="12" cy="12" r="9" strokeDasharray="34 20" />,
  file: (
    <>
      <polygon points="6,2 6,22 18,22 18,8 12,2" />
      <polyline points="12,2 12,8 18,8" />
    </>
  ),
  plus: (
    <>
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </>
  ),
};

/** Minimal hand-rolled outline icon set (no external icon library). */
function Icon({ name, size = 18, className = "" }) {
  const content = PATHS[name];
  if (!content) return null;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {content}
    </svg>
  );
}

export default Icon;
