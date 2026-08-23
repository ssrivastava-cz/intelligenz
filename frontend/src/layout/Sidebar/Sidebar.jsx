import { NavLink } from "react-router-dom";

import Icon from "../../components/Icon/Icon.jsx";
import { APP_NAME } from "../../constants/index.js";
import { NAV_ITEMS } from "../../config/navigation.js";
import "./Sidebar.css";

/**
 * Persistent left navigation: routed pages.
 */
function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__brand-mark">RTI</span>
        <span className="sidebar__brand-name">{APP_NAME}</span>
      </div>

      <nav className="sidebar__section" aria-label="Primary">
        <p className="sidebar__section-title">Navigation</p>
        <ul className="sidebar__nav-list">
          {NAV_ITEMS.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                end={item.path === "/"}
                className={({ isActive }) => `sidebar__nav-link ${isActive ? "sidebar__nav-link--active" : ""}`}
              >
                <Icon name={item.icon} size={17} />
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}

export default Sidebar;
