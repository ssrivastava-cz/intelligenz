import { useState } from "react";

import EmptyState from "../../components/EmptyState/EmptyState.jsx";
import { useKnowledgeAssistant } from "../../hooks/useKnowledgeAssistant.js";
import AnswerCard from "./components/AnswerCard.jsx";
import AskQuestionCard from "./components/AskQuestionCard.jsx";
import FeedbackModal from "./components/FeedbackModal.jsx";
import RecentQuestionsSection from "./components/RecentQuestionsSection.jsx";
import "./KnowledgeAssistant.css";

/**
 * Ask-a-question workspace for product docs, policies, and internal
 * guides — backed by the real Knowledge Assistant backend
 * (`useKnowledgeAssistant`, `POST /knowledge-assistant/ask` and
 * `POST /knowledge-assistant/feedback`). `generationId` always comes
 * from the backend; this page never generates one.
 */
function KnowledgeAssistant() {
  const [question, setQuestion] = useState("");
  const [feature, setFeature] = useState("");
  const [isFeedbackModalOpen, setIsFeedbackModalOpen] = useState(false);
  const { status, error, currentAnswer, recentQuestions, ask, recordFeedback } = useKnowledgeAssistant();

  function handleAsk() {
    ask(question, feature);
  }

  function handleHelpful() {
    if (!currentAnswer) return;
    recordFeedback(currentAnswer.generationId, {
      generationId: currentAnswer.generationId,
      evaluation: "GOOD",
      reason: null,
      description: null,
    });
  }

  async function handleFeedbackSubmit({ reason, description }) {
    if (!currentAnswer) return;
    await recordFeedback(currentAnswer.generationId, {
      generationId: currentAnswer.generationId,
      evaluation: "BAD",
      reason,
      description,
    });
    setIsFeedbackModalOpen(false);
  }

  return (
    <div className="knowledge-assistant">
      <AskQuestionCard
        question={question}
        onQuestionChange={setQuestion}
        feature={feature}
        onFeatureChange={setFeature}
        onAsk={handleAsk}
        isLoading={status === "loading"}
      />

      {status === "error" && (
        <EmptyState icon="x" title="Couldn't get an answer" description={error} />
      )}

      {currentAnswer && (
        <AnswerCard
          record={currentAnswer}
          onHelpful={handleHelpful}
          onNotHelpful={() => setIsFeedbackModalOpen(true)}
        />
      )}

      <RecentQuestionsSection items={recentQuestions} />

      <FeedbackModal
        isOpen={isFeedbackModalOpen}
        onCancel={() => setIsFeedbackModalOpen(false)}
        onSubmit={handleFeedbackSubmit}
      />
    </div>
  );
}

export default KnowledgeAssistant;
