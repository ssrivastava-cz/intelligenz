import Icon from "../Icon/Icon.jsx";
import "./EmptyState.css";

/**
 * Centered placeholder used for idle/empty panels.
 */
function EmptyState({ icon = "flask", title, description }) {
  return (
    <div className="empty-state">
      <div className="empty-state__icon">
        <Icon name={icon} size={28} />
      </div>
      <h3 className="empty-state__title">{title}</h3>
      {description && <p className="empty-state__description">{description}</p>}
    </div>
  );
}

export default EmptyState;
