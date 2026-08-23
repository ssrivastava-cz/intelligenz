import "./TextArea.css";

/**
 * Labeled textarea with an optional footer slot (e.g. a word counter).
 */
function TextArea({ id, label, value, onChange, placeholder, rows = 5, footer }) {
  return (
    <div className="text-area-field">
      {label && (
        <label className="text-area-field__label" htmlFor={id}>
          {label}
        </label>
      )}
      <textarea
        id={id}
        className="text-area-field__control"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
      />
      {footer && <div className="text-area-field__footer">{footer}</div>}
    </div>
  );
}

export default TextArea;
