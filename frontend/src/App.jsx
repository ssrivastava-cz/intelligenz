import { BrowserRouter, Route, Routes } from "react-router-dom";

import AppLayout from "./layout/AppLayout/AppLayout.jsx";
import GenerationHistory from "./pages/GenerationHistory/GenerationHistory.jsx";
import Home from "./pages/Home/Home.jsx";
import KnowledgeAssistant from "./pages/KnowledgeAssistant/KnowledgeAssistant.jsx";
import NotFound from "./pages/NotFound/NotFound.jsx";
import TestPlanGenerator from "./pages/TestPlanGenerator/TestPlanGenerator.jsx";
import UsageDashboard from "./pages/UsageDashboard/UsageDashboard.jsx";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Home />} />
          <Route path="knowledge-assistant" element={<KnowledgeAssistant />} />
          <Route path="test-plan-generator" element={<TestPlanGenerator />} />
          <Route path="generation-history" element={<GenerationHistory />} />
          <Route path="usage" element={<UsageDashboard />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
