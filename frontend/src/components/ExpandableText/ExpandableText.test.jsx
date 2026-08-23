import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import ExpandableText from "./ExpandableText.jsx";

describe("ExpandableText", () => {
  it("renders the full text and starts collapsed", () => {
    render(<ExpandableText text="Line one.\nLine two." />);

    expect(screen.getByText(/Line one/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show More" })).toBeInTheDocument();
  });

  it("expands and collapses when the toggle is clicked", async () => {
    const user = userEvent.setup();
    render(<ExpandableText text="Some long description text." />);

    await user.click(screen.getByRole("button", { name: "Show More" }));
    expect(screen.getByRole("button", { name: "Show Less" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show Less" }));
    expect(screen.getByRole("button", { name: "Show More" })).toBeInTheDocument();
  });

  it("never truncates the underlying text content, even while collapsed", () => {
    const longText = Array.from({ length: 20 }, (_, i) => `Line ${i + 1}.`).join("\n");
    render(<ExpandableText text={longText} />);

    expect(screen.getByText((_, element) => element?.textContent === longText)).toBeInTheDocument();
  });
});
