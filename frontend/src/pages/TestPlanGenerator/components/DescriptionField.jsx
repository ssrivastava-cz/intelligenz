import TextArea from "../../../components/TextArea/TextArea.jsx";
import { useWordCount } from "../../../hooks/useWordCount.js";

const WORD_LIMIT = 1500;

/** Optional free-text description with a live word counter. */
function DescriptionField({ value, onChange }) {
  const { wordCount, isOverLimit } = useWordCount(value, WORD_LIMIT);

  return (
    <TextArea
      id="description"
      label="Optional Description"
      value={value}
      onChange={onChange}
      placeholder="Add any extra context for the generator…"
      rows={6}
      footer={
        <span className={isOverLimit ? "text-area-field__footer--over-limit" : ""}>
          {wordCount} / {WORD_LIMIT} words
        </span>
      }
    />
  );
}

export default DescriptionField;
