import Icon from "../Icon/Icon.jsx";
import "./Select.css";

/**
 * Labeled native select styled to match the dark theme.
 * @param {{options: {value: string, label: string}[]}} props
 */
function Select({ id, label, value, onChange, options, placeholder = "Select…", required = false }) {
  return (
    <div className="select-field">
      {label && (
        <label className="select-field__label" htmlFor={id}>
          {label}
          {required && <span className="select-field__required">*</span>}
        </label>
      )}
      <div className="select-field__wrapper">
        <select id={id} className="select-field__control" value={value} onChange={(e) => onChange(e.target.value)}>
          <option value="" disabled>
            {placeholder}
          </option>
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <Icon name="chevronDown" size={16} className="select-field__chevron" />
      </div>
    </div>
  );
}

export default Select;
