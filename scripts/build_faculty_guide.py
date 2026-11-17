"""Render the editable faculty manual as a bookmarked, paginated academic PDF."""

import html
import json
import re
import textwrap
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/faculty_demonstration_guide.md"
OUTPUT = ROOT / "outputs/SalesVision_Faculty_Demonstration_and_User_Guide.pdf"
NAVY = colors.HexColor("#0F1D35")
BLUE = colors.HexColor("#245BEC")
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle("GuideBody", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8))
STYLES.add(ParagraphStyle("Script", parent=STYLES["GuideBody"], backColor=colors.HexColor("#EDF3FF"), borderPadding=8, spaceBefore=8, spaceAfter=14))
STYLES.add(ParagraphStyle("Cell", fontSize=8.3, leading=11))
STYLES.add(ParagraphStyle("CodeBlock", fontName="Courier", fontSize=8, leading=11, backColor=colors.HexColor("#F1F5F9"), borderPadding=8, spaceAfter=12))
for name in ("Heading1", "Heading2", "Heading3"):
    STYLES[name].textColor = NAVY
    STYLES[name].spaceBefore = 14
    STYLES[name].spaceAfter = 9


def markup(text):
    value = html.escape(text)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    return value.replace("- [ ]", "[  ]")


class GuideDocument(BaseDocTemplate):
    """Publish real destination page numbers in the multi-pass table of contents."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Heading1", "Heading2"}:
            level = 0 if flowable.style.name == "Heading1" else 1
            text = flowable.getPlainText()
            key = f"section-{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, text, self.page, key))


def page_furniture(canvas, document):
    width, height = canvas._pagesize
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(42, height - 27, "SalesVision | Sales Forecasting Using Python")
    canvas.setStrokeColor(BLUE)
    canvas.line(42, height - 34, width - 42, height - 34)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(42, 24, "Review 3 | Faculty demonstration and user guide")
    canvas.drawRightString(width - 42, 24, f"Page {document.page}")
    canvas.restoreState()


def render():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = GuideDocument(str(OUTPUT), pagesize=A4, title="SalesVision Faculty Demonstration and User Guide", author="SalesVision", leftMargin=42, rightMargin=42, topMargin=50, bottomMargin=44)
    templates = []
    for name, size in [("normal", A4), ("screenshot", landscape(A4))]:
        width, height = size
        templates.append(PageTemplate(id=name, pagesize=size, frames=[Frame(42, 44, width - 84, height - 96, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_furniture))
    doc.addPageTemplates(templates)
    cover = ParagraphStyle("Cover", fontName="Helvetica-Bold", fontSize=32, leading=40, textColor=NAVY)
    story = [Spacer(1, 95), Paragraph("SalesVision", cover), Spacer(1, 18), Paragraph("Sales Forecasting Using Python", STYLES["Title"]), Spacer(1, 20), Paragraph("Faculty Demonstration<br/>and User Guide", ParagraphStyle("Subtitle", fontSize=22, leading=29, textColor=BLUE, alignment=TA_CENTER)), Spacer(1, 35), Paragraph("University Minor Project • Review 3<br/>Windows 11 beginner edition", STYLES["GuideBody"]), Paragraph("Complete installation, twelve-workspace operating manual, speaking script, model interpretation, 52 viva answers, checklists and genuine screenshot evidence.", STYLES["GuideBody"]), Paragraph("Evidence basis: November 16, 2026 local audit. Gemini generation last reported service unavailability; labelled Local analytics remains available. Archived future dates are not predictions for today.", STYLES["Script"]), PageBreak(), Paragraph("Contents", STYLES["Title"])]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("TOCMain", fontSize=11, leading=16, spaceBefore=7), ParagraphStyle("TOCSub", fontSize=9, leading=13, leftIndent=14)]
    story.extend([toc, PageBreak()])
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = next(index for index, line in enumerate(lines) if line.startswith("## 1."))
    screenshots = []
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("# "):
            continue
        if line.startswith("```"):
            code = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.extend(textwrap.wrap(lines[index], width=91, replace_whitespace=False, drop_whitespace=False) or [""])
                index += 1
            index += 1
            story.append(Preformatted("\n".join(code), STYLES["CodeBlock"]))
        elif line.startswith("|"):
            rows = [line]
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1
            values = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows if not re.fullmatch(r"[| :\-]+", row)]
            count = len(values[0])
            width = A4[0] - 84
            weights = [1] * count
            if count == 2:
                weights = [1, 2.3]
            table = Table([[Paragraph(markup(cell), STYLES["Cell"]) for cell in row] for row in values], colWidths=[width * weight / sum(weights) for weight in weights], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE8FF")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCD6E3")), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
            story.extend([table, Spacer(1, 12)])
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*)\]\((.*)\)", line)
            caption, relative = match.groups()
            image_path = SOURCE.parent / relative
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if len(story) >= 2 and isinstance(story[-1], PageBreak) and isinstance(story[-2], NextPageTemplate):
                del story[-2:]
            story.extend([NextPageTemplate("screenshot"), PageBreak(), Paragraph(caption, STYLES["Heading2"])])
            image = Image(str(image_path))
            factor = min((landscape(A4)[0] - 84) / image.imageWidth, (landscape(A4)[1] - 155) / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * factor, image.imageHeight * factor
            story.extend([image, Paragraph("Genuine November 16, 2026 Chrome audit capture; visible page region, not a mock-up. Zoom the PDF for smaller interface labels.", STYLES["Cell"]), NextPageTemplate("normal"), PageBreak()])
            screenshots.append(relative)
        elif line.startswith("## "):
            if story and not isinstance(story[-1], PageBreak):
                story.append(PageBreak())
            story.append(Paragraph(markup(line[3:]), STYLES["Heading1"]))
        elif line.startswith("### "):
            story.append(Paragraph(markup(line[4:]), STYLES["Heading2"]))
        else:
            story.append(Paragraph(markup(line), STYLES["Script"] if line.startswith("**Say:") else STYLES["GuideBody"]))
    doc.multiBuild(story)
    reader = PdfReader(OUTPUT)
    texts = [page.extract_text() for page in reader.pages]
    required = ["1. Project orientation", "2. Windows 11", "3. Dataset", "4. Complete application", "5. Models", "6. Evaluation", "7. Review 3", "8. Faculty questions", "9. Review 3", "10. Demonstration timelines", "11. Genuine screenshot", "12. Final quick reference", "Q52."]
    joined = "\n".join(texts)
    if not all(section in joined for section in required):
        raise ValueError("A required guide section is missing from the rendered PDF.")
    if not all(f"Page {number}" in text for number, text in enumerate(texts, 1)):
        raise ValueError("PDF page numbers are incomplete.")
    result = {"status": "passed", "pdf": OUTPUT.relative_to(ROOT).as_posix(), "bytes": OUTPUT.stat().st_size, "pages": len(texts), "screenshots": len(screenshots), "chapters": 12, "questions": 52, "page_numbers": "passed", "text_sections": "passed"}
    evidence = ROOT / "outputs/qa/faculty-guide-verification.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    render()
