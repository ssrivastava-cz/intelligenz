import Icon from "../Icon/Icon.jsx";
import "./Drawer.css";

/**
 * Generic slide-in side panel with a backdrop, title, and close button.
 * Renders nothing when closed.
 * @param {{isOpen: boolean, title: string, onClose: () => void, children?: React.ReactNode}} props
 */
function Drawer({ isOpen, title, onClose, children }) {
  if (!isOpen) return null;

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="drawer__header">
          <h3 className="drawer__title">{title}</h3>
          <button type="button" className="drawer__close" onClick={onClose} aria-label="Close">
            <Icon name="x" size={18} />
          </button>
        </header>
        <div className="drawer__body">{children}</div>
      </aside>
    </div>
  );
}

export default Drawer;
