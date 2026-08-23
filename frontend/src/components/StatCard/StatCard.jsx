import Icon from "../Icon/Icon.jsx";
import "./StatCard.css";

/**
 * Single metric tile used on the Usage Dashboard.
 * @param {{label: string, value: string, hint?: string, icon?: string}} props
 */
function StatCard({ label, value, hint, icon = "chart" }) {
  return (
    <div className="stat-card">
      <div className="stat-card__icon">
        <Icon name={icon} size={18} />
      </div>
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__value">{value}</div>
      {hint && <div className="stat-card__hint">{hint}</div>}
    </div>
  );
}

export default StatCard;
