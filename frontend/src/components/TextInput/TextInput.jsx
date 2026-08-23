import "./TextInput.css";

/**
 * Labeled single-line text input.
 */
function TextInput({ id, label, value, onChange, placeholder, hint, required = false }) {
  return (
    <div className="text-input-field">
      {label && (
        <label className="text-input-field__label" htmlFor={id}>
          {label}
          {required && <span className="text-input-field__required">*</span>}
        </label>
      )}
      <input
        id={id}
        type="text"
        className="text-input-field__control"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
      {hint && <p className="text-input-field__hint">{hint}</p>}
    </div>
  );
}

export default TextInput;
