import { Link } from "react-router-dom";

import Badge from "../../components/Badge/Badge.jsx";
import Card from "../../components/Card/Card.jsx";
import Icon from "../../components/Icon/Icon.jsx";
import { COMING_SOON_ITEMS, NAV_ITEMS } from "../../config/navigation.js";
import "./Home.css";

const FEATURE_CARDS = NAV_ITEMS.filter((item) => item.path !== "/");

/**
 * Landing page: quick links into the available tools plus a preview
 * of what's on the roadmap.
 */
function Home() {
  return (
    <div className="home">
      <section className="home__hero">
        <h2 className="home__hero-title">Welcome to Release Team Intelligenz</h2>
        <p className="home__hero-description">
          An AI-powered engineering assistant for QA teams — generate standardized test plans from Redmine
          tickets, workflow docs, and historical test data.
        </p>
      </section>

      <section className="home__section">
        <h3 className="home__section-title">Available Tools</h3>
        <div className="home__grid">
          {FEATURE_CARDS.map((item) => (
            <Link key={item.path} to={item.path} className="home__feature-card">
              <div className="home__feature-icon">
                <Icon name={item.icon} size={20} />
              </div>
              <span className="home__feature-label">{item.label}</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="home__section">
        <Card title="On the Roadmap" subtitle="Additional AI-assisted tools planned for future releases">
          <div className="home__coming-soon-list">
            {COMING_SOON_ITEMS.map((label) => (
              <div key={label} className="home__coming-soon-item">
                <span>{label}</span>
                <Badge>Soon</Badge>
              </div>
            ))}
          </div>
        </Card>
      </section>
    </div>
  );
}

export default Home;
