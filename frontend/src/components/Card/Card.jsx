import "./Card.css";

/**
 * Generic elevated surface with an optional header (title/subtitle/actions).
 */
function Card({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={`card ${className}`.trim()}>
      {(title || actions) && (
        <header className="card__header">
          <div>
            {title && <h3 className="card__title">{title}</h3>}
            {subtitle && <p className="card__subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="card__actions">{actions}</div>}
        </header>
      )}
      <div className="card__body">{children}</div>
    </section>
  );
}

export default Card;
