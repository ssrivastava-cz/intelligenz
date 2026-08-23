import "./Badge.css";

/**
 * Small status/tag pill.
 * @param {{tone?: "default"|"success"|"warning"|"danger"|"info"}} props
 */
function Badge({ children, tone = "default", className = "" }) {
  return <span className={`badge badge--${tone} ${className}`.trim()}>{children}</span>;
}

export default Badge;
