import ReactMarkdown from "react-markdown";
import "./MarkdownText.css";

/**
 * Renders a Markdown-formatted answer (bold, headings, numbered/bullet
 * lists, paragraphs) using the app's existing typography. `react-markdown`
 * only ever produces the fixed set of elements Markdown syntax maps to —
 * it never renders raw HTML embedded in `text` (no `rehype-raw` plugin
 * is used here), so any `<script>`/HTML in a generated answer is
 * neutralized rather than executed. `className` lets each caller keep
 * its own existing color/size styling (e.g. `answer-card__answer`);
 * this component only adds spacing/list/heading rules on top of it.
 * @param {{text: string, className?: string}} props
 */
function MarkdownText({ text, className = "" }) {
  return (
    <div className={["markdown-text", className].filter(Boolean).join(" ")}>
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );
}

export default MarkdownText;
