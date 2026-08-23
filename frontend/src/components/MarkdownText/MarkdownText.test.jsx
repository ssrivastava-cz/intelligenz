import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import MarkdownText from "./MarkdownText.jsx";

describe("MarkdownText", () => {
  it("renders **bold** markdown as a real <strong> element", () => {
    const { container } = render(<MarkdownText text="The appointment is marked **eligible**." />);

    const strong = container.querySelector("strong");
    expect(strong).toBeInTheDocument();
    expect(strong).toHaveTextContent("eligible");
    // The literal ** markers are gone, not shown as text.
    expect(container.textContent).not.toContain("**");
  });

  it("renders a numbered list as an ordered list of items", () => {
    const text = "1. Identify the appointment\n2. Evaluate eligibility\n3. Resolve required information";
    const { container } = render(<MarkdownText text={text} />);

    const list = container.querySelector("ol");
    expect(list).toBeInTheDocument();
    const items = list.querySelectorAll("li");
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("Identify the appointment");
    expect(items[2]).toHaveTextContent("Resolve required information");
  });

  it("renders a bullet list as an unordered list of items", () => {
    const text = "- Patient\n- Plan\n- Site\n- Provider";
    const { container } = render(<MarkdownText text={text} />);

    const list = container.querySelector("ul");
    expect(list).toBeInTheDocument();
    const items = [...list.querySelectorAll("li")];
    expect(items).toHaveLength(4);
    expect(items.map((item) => item.textContent)).toEqual(
      expect.arrayContaining(["Patient", "Plan", "Site", "Provider"]),
    );
  });

  it("renders a ### heading as a real heading element", () => {
    const { container } = render(<MarkdownText text="### Important limitation\n\nSome detail." />);

    const heading = container.querySelector("h3");
    expect(heading).toBeInTheDocument();
    expect(heading).toHaveTextContent("Important limitation");
    expect(container.textContent).not.toContain("###");
  });

  it("renders separate paragraphs as separate <p> elements, not one run-on block", () => {
    const text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
    const { container } = render(<MarkdownText text={text} />);

    const paragraphs = container.querySelectorAll("p");
    expect(paragraphs).toHaveLength(3);
    expect(paragraphs[0]).toHaveTextContent("First paragraph.");
    expect(paragraphs[1]).toHaveTextContent("Second paragraph.");
    expect(paragraphs[2]).toHaveTextContent("Third paragraph.");
  });

  it("renders a plain answer with no Markdown syntax as ordinary readable text", () => {
    render(<MarkdownText text="Cost is calculated from prompt and completion tokens." />);

    expect(screen.getByText("Cost is calculated from prompt and completion tokens.")).toBeInTheDocument();
  });

  it("never renders embedded HTML/script content as real elements — it's neutralized, not executed", () => {
    const text = "Here is the answer.\n\n<script>window.__xss = true;</script><img src=x onerror=\"window.__xss = true\">";
    const { container } = render(<MarkdownText text={text} />);

    expect(container.querySelector("script")).not.toBeInTheDocument();
    expect(container.querySelector("img")).not.toBeInTheDocument();
    expect(container.querySelector("[onerror]")).not.toBeInTheDocument();
    expect(window.__xss).toBeUndefined();
    // Neutralized as inert escaped text, never a real <script> tag.
    expect(container.innerHTML).not.toContain("<script>");
    expect(container.innerHTML).toContain("&lt;script&gt;");
  });

  it("applies the caller's className to the wrapping element for existing styling", () => {
    const { container } = render(<MarkdownText text="Some answer." className="answer-card__answer" />);

    expect(container.querySelector(".answer-card__answer")).toBeInTheDocument();
  });
});
