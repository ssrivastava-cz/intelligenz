import { TEST_PLAN_COLUMNS } from "../../../config/testPlanColumns.js";
import TestPlanRow from "./TestPlanRow.jsx";
import "./TestPlanTable.css";

/** Editable table of generated test cases. */
function TestPlanTable({ rows, onUpdateRow, onDeleteRow }) {
  return (
    <div className="test-plan-table-wrap">
      <table className="test-plan-table">
        <thead>
          <tr>
            <th className="test-plan-table__expand-col" aria-hidden="true" />
            {TEST_PLAN_COLUMNS.map((col) => (
              <th key={col.key}>{col.label}</th>
            ))}
            <th className="test-plan-table__actions-col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <TestPlanRow
              key={row.id}
              row={row}
              onUpdate={(patch) => onUpdateRow(row.id, patch)}
              onDelete={() => onDeleteRow(row.id)}
            />
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={TEST_PLAN_COLUMNS.length + 2} className="test-plan-table__empty">
                No test cases yet. Click "Add Row" to create one.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default TestPlanTable;
