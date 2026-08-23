import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import Table from "./Table.jsx";

const COLUMNS = [
  { key: "name", label: "Name" },
  { key: "count", label: "Count", render: (row) => `${row.count} items` },
];

const ROWS = [
  { id: "a", name: "Alpha", count: 3 },
  { id: "b", name: "Beta", count: 5 },
];

describe("Table", () => {
  it("renders a header row from the columns config", () => {
    render(<Table columns={COLUMNS} rows={ROWS} getRowKey={(row) => row.id} />);

    expect(screen.getByRole("columnheader", { name: "Name" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Count" })).toBeInTheDocument();
  });

  it("renders one row per item, using the default cell value when no render is given", () => {
    render(<Table columns={COLUMNS} rows={ROWS} getRowKey={(row) => row.id} />);

    expect(screen.getByText("Alpha")).toBeInTheDocument();
    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("uses a column's render function for custom cell content", () => {
    render(<Table columns={COLUMNS} rows={ROWS} getRowKey={(row) => row.id} />);

    expect(screen.getByText("3 items")).toBeInTheDocument();
    expect(screen.getByText("5 items")).toBeInTheDocument();
  });

  it("renders no data rows when given an empty array", () => {
    render(<Table columns={COLUMNS} rows={[]} getRowKey={(row) => row.id} />);

    expect(screen.queryAllByRole("row")).toHaveLength(1); // header row only
  });
});
