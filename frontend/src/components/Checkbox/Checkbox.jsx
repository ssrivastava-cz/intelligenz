import "./Checkbox.css";

/**
 * Labeled checkbox with an optional helper description.
 */
function Checkbox({ id, label, description, checked, onChange }) {
  return (
    <label className="checkbox" htmlFor={id}>
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="checkbox__box" aria-hidden="true" />
      <span className="checkbox__text">
        <span className="checkbox__label">{label}</span>
        {description && <span className="checkbox__description">{description}</span>}
      </span>
    </label>
  );
}

export default Checkbox;
