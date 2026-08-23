import { useState } from "react";

import Badge from "../../../components/Badge/Badge.jsx";
import Button from "../../../components/Button/Button.jsx";
import Icon from "../../../components/Icon/Icon.jsx";
import { PRIORITY_OPTIONS, TEST_PLAN_COLUMNS, TYPE_OPTIONS } from "../../../config/testPlanColumns.js";
import "./TestPlanRow.css";

const PRIORITY_TONE = { High: "danger", Medium: "warning", Low: "info" };

function EditableCell({ col, value, onChange }) {
  if (col.key === "priority" || col.key === "type") {
    const options = col.key === "priority" ? PRIORITY_OPTIONS : TYPE_OPTIONS;
    return (
      <select className="test-plan-row__input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    );
  }

  if (col.key === "automationCandidate") {
    return <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />;
  }

  if (col.key === "steps") {
    return (
      <textarea
        className="test-plan-row__input test-plan-row__input--textarea"
        rows={3}
        value={value.join("\n")}
        onChange={(e) => onChange(e.target.value.split("\n").filter(Boolean))}
      />
    );
  }

  if (["preconditions", "testCase", "expectedResult"].includes(col.key)) {
    return (
      <textarea
        className="test-plan-row__input test-plan-row__input--textarea"
        rows={2}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  return <input className="test-plan-row__input" type="text" value={value} onChange={(e) => onChange(e.target.value)} />;
}

function DisplayCell({ col, value }) {
  if (col.key === "priority") return <Badge tone={PRIORITY_TONE[value] ?? "default"}>{value}</Badge>;
  if (col.key === "type") return <Badge>{value}</Badge>;
  if (col.key === "automationCandidate") return <Badge tone={value ? "success" : "default"}>{value ? "Yes" : "No"}</Badge>;
  if (col.key === "steps") return <span className="test-plan-row__truncate">{value.join(" → ")}</span>;
  return <span className="test-plan-row__truncate">{value}</span>;
}

/** Single test-case row: view mode, inline edit mode, and an expandable detail row. */
function TestPlanRow({ row, onUpdate, onDelete }) {
  const [isEditing, setIsEditing] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [draft, setDraft] = useState(row);

  function startEditing() {
    setDraft(row);
    setIsEditing(true);
  }

  function save() {
    onUpdate(draft);
    setIsEditing(false);
  }

  function cancel() {
    setIsEditing(false);
  }

  return (
    <>
      <tr className={isEditing ? "test-plan-row--editing" : ""}>
        <td>
          <button
            type="button"
            className="test-plan-row__expand-btn"
            onClick={() => setIsExpanded((prev) => !prev)}
            aria-label={isExpanded ? "Collapse row" : "Expand row"}
          >
            <Icon name={isExpanded ? "chevronDown" : "chevronRight"} size={14} />
          </button>
        </td>
        {TEST_PLAN_COLUMNS.map((col) => (
          <td key={col.key}>
            {isEditing ? (
              <EditableCell col={col} value={draft[col.key]} onChange={(value) => setDraft({ ...draft, [col.key]: value })} />
            ) : (
              <DisplayCell col={col} value={row[col.key]} />
            )}
          </td>
        ))}
        <td className="test-plan-row__actions">
          {isEditing ? (
            <>
              <Button variant="primary" size="sm" onClick={save}>
                Save
              </Button>
              <Button variant="ghost" size="sm" onClick={cancel}>
                Cancel
              </Button>
            </>
          ) : (
            <>
              <Button variant="ghost" size="sm" onClick={startEditing}>
                Edit
              </Button>
              <Button variant="danger" size="sm" onClick={onDelete}>
                Delete
              </Button>
            </>
          )}
        </td>
      </tr>

      {isExpanded && (
        <tr className="test-plan-row__detail">
          <td colSpan={TEST_PLAN_COLUMNS.length + 2}>
            <div className="test-plan-row__detail-grid">
              <div>
                <h4>Preconditions</h4>
                <p>{row.preconditions}</p>
              </div>
              <div>
                <h4>Steps</h4>
                <ol>
                  {row.steps.map((step, index) => (
                    <li key={`${row.id}-step-${index}`}>{step}</li>
                  ))}
                </ol>
              </div>
              <div>
                <h4>Expected Result</h4>
                <p>{row.expectedResult}</p>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default TestPlanRow;
