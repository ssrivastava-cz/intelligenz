import { useLocation } from "react-router-dom";

import { PAGE_TITLES } from "../../config/navigation.js";
import "./TopNavbar.css";

/**
 * Top bar showing the current page title.
 */
function TopNavbar() {
  const { pathname } = useLocation();
  const title = PAGE_TITLES[pathname] ?? "Not Found";

  return (
    <header className="top-navbar">
      <h1 className="top-navbar__title">{title}</h1>
    </header>
  );
}

export default TopNavbar;
