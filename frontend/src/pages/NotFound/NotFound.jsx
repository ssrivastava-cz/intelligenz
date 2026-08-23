import { Link } from "react-router-dom";

import Button from "../../components/Button/Button.jsx";
import "./NotFound.css";

/** Catch-all route for unmatched paths. */
function NotFound() {
  return (
    <div className="not-found">
      <span className="not-found__code">404</span>
      <h2 className="not-found__title">Page not found</h2>
      <p className="not-found__description">The page you're looking for doesn't exist or has moved.</p>
      <Link to="/">
        <Button variant="primary">Back to Dashboard</Button>
      </Link>
    </div>
  );
}

export default NotFound;
