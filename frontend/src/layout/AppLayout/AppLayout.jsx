import { Outlet } from "react-router-dom";

import Sidebar from "../Sidebar/Sidebar.jsx";
import TopNavbar from "../TopNavbar/TopNavbar.jsx";
import "./AppLayout.css";

/**
 * Application shell: persistent sidebar + top navbar around a
 * responsive, routed content area.
 */
function AppLayout() {
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="app-layout__main">
        <TopNavbar />
        <main className="app-layout__content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default AppLayout;
