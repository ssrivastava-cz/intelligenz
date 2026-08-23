import "./Button.css";

/**
 * Reusable button with a small set of visual variants.
 * @param {{variant?: "primary"|"secondary"|"ghost"|"danger", size?: "sm"|"md", fullWidth?: boolean}} props
 */
function Button({
  children,
  variant = "primary",
  size = "md",
  fullWidth = false,
  className = "",
  type = "button",
  ...rest
}) {
  const classes = ["btn", `btn--${variant}`, `btn--${size}`, fullWidth ? "btn--full" : "", className]
    .filter(Boolean)
    .join(" ");

  return (
    <button type={type} className={classes} {...rest}>
      {children}
    </button>
  );
}

export default Button;
