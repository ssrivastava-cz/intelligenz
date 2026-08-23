import "./Table.css";

/**
 * Generic read-only data table: pass `columns` (each with a `key`,
 * `label`, and optional `render(row)` for custom cell content) and
 * `rows`. Keeps every usage/history table in the app rendering the
 * same markup/styling instead of each screen hand-rolling its own
 * <table>.
 * @param {{columns: {key: string, label: string, render?: (row: object) => React.ReactNode}[], rows: object[], getRowKey: (row: object) => string}} props
 */
function Table({ columns, rows, getRowKey }) {
  return (
    <div className="data-table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={getRowKey(row)}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row) : row[column.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Table;
