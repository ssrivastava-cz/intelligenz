/** Primary sidebar navigation — routed pages. */
export const NAV_ITEMS = [
  { label: "Dashboard", path: "/", icon: "dashboard" },
  { label: "Knowledge Assistant", path: "/knowledge-assistant", icon: "search" },
  { label: "AI Test Plan Generator", path: "/test-plan-generator", icon: "flask" },
  { label: "Test Plan Generation History", path: "/generation-history", icon: "clock" },
  { label: "Usage", path: "/usage", icon: "chart" },
];

/** Roadmap items shown in the sidebar but not yet navigable. */
export const COMING_SOON_ITEMS = ["Cozeva Knowledge Assistant"];

/** Maps a route path to the title shown in the top navigation bar. */
export const PAGE_TITLES = {
  "/": "Dashboard",
  "/knowledge-assistant": "Knowledge Assistant",
  "/test-plan-generator": "AI Test Plan Generator",
  "/generation-history": "Test Plan Generation History",
  "/usage": "Usage Dashboard",
};
