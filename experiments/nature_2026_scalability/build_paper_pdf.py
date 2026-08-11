"""Render the paper report as a publication-style PDF without LibreOffice.

This is a deterministic ReportLab rendering of the final DOCX's ordered content.
It is used on hosts where neither LibreOffice nor Microsoft Word is installed.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)


NAVY = colors.HexColor("#13243A")
BLUE = colors.HexColor("#2E74B5")
DARK_BLUE = colors.HexColor("#1F4D78")
INK = colors.HexColor("#253140")
MUTED = colors.HexColor("#66717E")
GRID = colors.HexColor("#D9E0E8")
PALE = colors.HexColor("#F4F7FA")
RED = colors.HexColor("#A23A3A")
WHITE = colors.white


def iter_blocks(document: Document):
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, document)


def paragraph_markup(paragraph: DocxParagraph) -> str:
    parts: list[str] = []
    for run in paragraph.runs:
        value = html.escape(run.text).replace("\n", "<br/>")
        if not value:
            continue
        if run.bold:
            value = f"<b>{value}</b>"
        if run.italic:
            value = f"<i>{value}</i>"
        parts.append(value)
    return "".join(parts) or html.escape(paragraph.text)


def has_page_break(paragraph: DocxParagraph) -> bool:
    return bool(paragraph._p.xpath('.//w:br[@w:type="page"]'))


def has_image(paragraph: DocxParagraph) -> bool:
    return bool(paragraph._p.xpath(".//w:drawing"))


def has_shading(paragraph: DocxParagraph) -> bool:
    return bool(paragraph._p.xpath("./w:pPr/w:shd"))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("PaperBody", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.4, leading=12.5, textColor=INK,
                               alignment=TA_JUSTIFY, spaceAfter=7),
        "title": ParagraphStyle("PaperTitle", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=23, leading=28, textColor=NAVY,
                                alignment=TA_CENTER, spaceAfter=9),
        "subtitle": ParagraphStyle("PaperSubtitle", parent=base["Normal"], fontName="Helvetica",
                                   fontSize=12, leading=15, textColor=DARK_BLUE,
                                   alignment=TA_CENTER, spaceAfter=15),
        "h1": ParagraphStyle("PaperH1", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14.5, leading=18, textColor=BLUE,
                             spaceBefore=14, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("PaperH2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=11.5, leading=14, textColor=BLUE,
                             spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "h3": ParagraphStyle("PaperH3", parent=base["Heading3"], fontName="Helvetica-Bold",
                             fontSize=10.5, leading=13, textColor=DARK_BLUE,
                             spaceBefore=7, spaceAfter=4, keepWithNext=True),
        "caption": ParagraphStyle("PaperCaption", parent=base["Normal"], fontName="Helvetica",
                                  fontSize=7.7, leading=9.5, textColor=INK,
                                  spaceBefore=4, spaceAfter=9),
        "meta": ParagraphStyle("PaperMeta", parent=base["Normal"], fontName="Helvetica",
                               fontSize=9.2, leading=11, textColor=MUTED,
                               alignment=TA_CENTER, spaceAfter=8),
        "kicker": ParagraphStyle("PaperKicker", parent=base["Normal"], fontName="Helvetica-Bold",
                                 fontSize=10, leading=12, textColor=colors.HexColor("#A57418"),
                                 alignment=TA_CENTER, spaceAfter=14),
        "reference": ParagraphStyle("PaperReference", parent=base["Normal"], fontName="Helvetica",
                                    fontSize=8.1, leading=10.2, textColor=INK,
                                    leftIndent=18, firstLineIndent=-18, spaceAfter=5),
        "bullet": ParagraphStyle("PaperBullet", parent=base["Normal"], fontName="Helvetica",
                                 fontSize=9.2, leading=11.8, textColor=INK,
                                 leftIndent=17, firstLineIndent=0, spaceAfter=4),
        "callout": ParagraphStyle("PaperCallout", parent=base["Normal"], fontName="Helvetica",
                                  fontSize=9.2, leading=12, textColor=INK,
                                  leftIndent=0, rightIndent=0, spaceAfter=0),
    }


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=letter, leftMargin=inch, rightMargin=inch,
                         topMargin=.78 * inch, bottomMargin=.72 * inch,
                         title="HDFA-RL versus Google RL comparative scientific report",
                         author="HDFA-RL Suite")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=self.decorate))

    def decorate(self, canvas, document):
        page = canvas.getPageNumber()
        canvas.saveState()
        if page > 1:
            canvas.setFont("Helvetica-Bold", 7.4)
            canvas.setFillColor(MUTED)
            canvas.drawString(inch, 10.54 * inch,
                              "HDFA-RL SUITE | COMPARATIVE SCIENTIFIC REPORT")
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(.5)
            canvas.line(inch, 10.47 * inch, 7.5 * inch, 10.47 * inch)
        canvas.setFont("Helvetica", 7.7)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(7.5 * inch, .42 * inch,
                               f"Nature 2026 comparison  |  {page}")
        canvas.restoreState()


def table_flowable(table: DocxTable) -> Table:
    data = []
    for row_index, row in enumerate(table.rows):
        converted = []
        for cell in row.cells:
            text = "\n".join(p.text for p in cell.paragraphs).strip()
            style = ParagraphStyle(
                f"Cell{row_index}", fontName="Helvetica-Bold" if row_index == 0 else "Helvetica",
                fontSize=6.7 if len(row.cells) >= 6 else 7.1,
                leading=8.2 if len(row.cells) >= 6 else 8.8,
                textColor=WHITE if row_index == 0 else INK,
            )
            converted.append(Paragraph(html.escape(text), style))
        data.append(converted)
    widths = []
    for cell in table.rows[0].cells:
        tc_w = cell._tc.xpath("./w:tcPr/w:tcW")
        widths.append(int(tc_w[0].get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w"))
                      if tc_w else 1)
    total = sum(widths)
    col_widths = [6.5 * inch * value / total for value in widths]
    flow = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("GRID", (0, 0), (-1, -1), .35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(1, len(data)):
        if row % 2 == 1:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    flow.setStyle(TableStyle(commands))
    return flow


def build(docx_path: Path, figure_dir: Path, output: Path) -> None:
    document = Document(docx_path)
    style = styles()
    blocks = list(iter_blocks(document))
    figure_paths = [
        figure_dir / "figure1-effectiveness-and-lifecycle.png",
        figure_dir / "figure2-primary-trajectories.png",
        figure_dir / "figure3-logical-and-acceptance.png",
        figure_dir / "figure4-google-paper-scalability-comparison.png",
        figure_dir / "figure5-computational-scaling.png",
    ]
    figure_index = 0
    story = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if isinstance(block, DocxTable):
            story.extend((table_flowable(block), Spacer(1, 6)))
            index += 1
            continue
        paragraph = block
        if has_page_break(paragraph):
            story.append(PageBreak())
            index += 1
            continue
        if has_image(paragraph):
            image = Image(str(figure_paths[figure_index]), width=6.5 * inch,
                          height=6.5 * inch * 1800 / 3200)
            image.hAlign = "CENTER"
            items = [image]
            if index + 1 < len(blocks) and isinstance(blocks[index + 1], DocxParagraph) \
                    and blocks[index + 1].style.name == "Caption":
                items.extend((Spacer(1, 3), Paragraph(paragraph_markup(blocks[index + 1]),
                                                       style["caption"])))
                index += 1
            story.append(KeepTogether(items))
            figure_index += 1
            index += 1
            continue
        value = paragraph_markup(paragraph)
        if not paragraph.text.strip():
            story.append(Spacer(1, 10))
            index += 1
            continue
        name = paragraph.style.name
        if has_shading(paragraph):
            callout = Table([[Paragraph(value, style["callout"])]], colWidths=[6.5 * inch])
            callout.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF3F8")
                 if "PRIMARY FINDING" not in paragraph.text else colors.HexColor("#F7EEF0")),
                ("BOX", (0, 0), (-1, -1), .4, GRID),
                ("LINEBEFORE", (0, 0), (0, -1), 3, RED if "PRIMARY FINDING" in paragraph.text else BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.extend((callout, Spacer(1, 7)))
        elif name == "Title":
            story.append(Paragraph(value, style["title"]))
        elif name == "Subtitle":
            story.append(Paragraph(value, style["subtitle"]))
        elif name == "Heading 1":
            story.append(Paragraph(value, style["h1"]))
        elif name == "Heading 2":
            story.append(Paragraph(value, style["h2"]))
        elif name == "Heading 3":
            story.append(Paragraph(value, style["h3"]))
        elif name == "Caption":
            story.append(Paragraph(value, style["caption"]))
        elif name == "List Bullet":
            item = ListItem(Paragraph(value, style["bullet"]), leftIndent=12)
            story.append(ListFlowable([item], bulletType="bullet", start="circle",
                                      leftIndent=18, bulletFontName="Helvetica",
                                      bulletFontSize=7, spaceAfter=2))
        elif paragraph.text.startswith("[") and paragraph.text[:3].rstrip("0123456789") == "[":
            story.append(Paragraph(value, style["reference"]))
        elif paragraph.alignment == 1:
            chosen = style["kicker"] if "COMPARATIVE QEC" in paragraph.text else style["meta"]
            story.append(Paragraph(value, chosen))
        else:
            story.append(Paragraph(value, style["body"]))
        index += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    PaperDocTemplate(str(output)).build(story)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.docx, args.figures, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
