import { useLocation } from "react-router-dom";

import { PAGE_TITLES } from "../../config/navigation.js";
import "./TopNavbar.css";

/**
 * Top bar showing the current page title and a placeholder user avatar.
 */
function TopNavbar() {
  const { pathname } = useLocation();
  const title = PAGE_TITLES[pathname] ?? "Not Found";

  return (
    <header className="top-navbar">
      <h1 className="top-navbar__title">{title}</h1>
      <div className="top-navbar__right">
        <span className="top-navbar__env-badge">Local Dev</span>
        <div className="top-navbar__avatar" aria-hidden="true">
          QA
        </div>
      </div>
    </header>
  );
}

export default TopNavbar;
