"""
Generates the LMS Submission Word (.docx) and PDF (.pdf) Documents for
Foundations of Generative AI (ICT4403) - Assignment 3 (Steps 3 & 4).

Outputs:
- KumarKshitij_230953488_FGAI_Assignment3.docx
- KumarKshitij_230953488_FGAI_Assignment3.pdf
"""

import os
import sys
import json


def load_results_summary():
    if os.path.exists("results/metrics_summary.json"):
        with open("results/metrics_summary.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def create_submission_word_doc(
    student_name: str = "Kumar Kshitij",
    registration_num: str = "230953488",
    github_link: str = "https://github.com/Kshiku12/Gen-AI-Assignment",
    output_filename: str = "KumarKshitij_230953488_FGAI_Assignment3.docx",
):
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    summary = load_results_summary()
    primary = summary["primary_models"] if summary else {}

    # Find best model by BLEU
    best_model_name = "Gemma-3-1B"
    best_bleu = 0.0
    best_chrf = 0.0
    if primary:
        best_model_name = max(primary.keys(), key=lambda k: primary[k]["bleu"])
        best_bleu = primary[best_model_name]["bleu"]
        best_chrf = primary[best_model_name]["chrf"]

    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    def set_cell_background(cell, hex_color):
        tcPr = cell._element.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)

    # Header
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(
        "Foundations of Generative AI (ICT4403)\nAssignment 3 Submission Document"
    )
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(19)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(31, 78, 121)

    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle_p.add_run(
        "Neural Machine Translation System (English to French) using Basic Transformer, mT5, LLaMA, and Gemma"
    )
    sub_run.font.name = "Calibri"
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(89, 89, 89)

    # Section 1: Submission Details Table (Step 3)
    h1 = doc.add_heading("1. Submission Details", level=2)
    h1.style.font.color.rgb = RGBColor(31, 78, 121)

    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    labels_data = [
        ("Student Name", student_name),
        ("Registration Number", registration_num),
        ("Assignment Title", "Foundations of Generative AI (ICT4403) - Assignment 3"),
        (
            "Title of Developed System",
            "Comparative English-to-French Neural Machine Translation System using Encoder-Decoder (Basic Transformer, mT5) and Decoder-Only (LLaMA, Gemma) Architectures",
        ),
        (
            "Dataset Name and Link",
            "Tatoeba / OPUS English-French Parallel Corpus\nhttps://www.manythings.org/anki/fra-eng.zip\nhttps://huggingface.co/datasets/Helsinki-NLP/opus-100",
        ),
        ("GitHub Repository Link", github_link),
    ]

    for i, (label, val) in enumerate(labels_data):
        row = table.rows[i]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        cell_lbl.width = Inches(2.2)
        cell_val.width = Inches(4.4)
        set_cell_background(cell_lbl, "F2F2F2")

        p_lbl = cell_lbl.paragraphs[0]
        r_lbl = p_lbl.add_run(label)
        r_lbl.font.bold = True
        r_lbl.font.size = Pt(10.5)
        r_lbl.font.name = "Calibri"

        p_val = cell_val.paragraphs[0]
        r_val = p_val.add_run(val)
        r_val.font.size = Pt(10.5)
        r_val.font.name = "Calibri"
        if "http" in val:
            r_val.font.color.rgb = RGBColor(5, 99, 193)
            r_val.font.underline = True

    # Section 2: Executive Summary & Quantitative Results
    h2 = doc.add_heading("2. Executive Summary & Quantitative Evaluation", level=2)
    h2.style.font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "This project develops, adapts, and evaluates an end-to-end English-to-French (en -> fr) Neural Machine Translation "
        "system across four distinct generative architectures: (1) a Basic Encoder-Decoder Transformer trained from scratch in PyTorch, "
        "(2) mT5-small (google/mt5-small, Encoder-Decoder) fine-tuned via supervised Seq2Seq learning, "
        "(3) LLaMA-3.2-1B-Instruct (Decoder-Only) adapted via both 3-Shot In-Context Prompting and LoRA Fine-Tuning, and "
        "(4) Gemma-3-1B-IT (Decoder-Only) adapted via both 3-Shot In-Context Prompting and LoRA Fine-Tuning. "
        "All models were evaluated on the identical held-out test set from the Tatoeba English-French Parallel Corpus "
        "(12,000 sentence pairs split into 9,600 train, 1,200 validation, and 1,200 test pairs)."
    )

    if summary and "all_configurations" in summary:
        all_cfg = summary["all_configurations"]
        order = [
            "Basic Transformer",
            "mT5-small",
            "LLaMA-3.2-1B (Prompting)",
            "LLaMA-3.2-1B",
            "Gemma-3-1B (Prompting)",
            "Gemma-3-1B",
        ]
        avail = [k for k in order if k in all_cfg]
        res_table = doc.add_table(rows=len(avail) + 1, cols=7)
        res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers = ["Model", "Architecture", "Adaptation", "BLEU", "chrF", "Train Time (s)", "Latency (ms/sent)"]
        for j, h_text in enumerate(headers):
            c = res_table.rows[0].cells[j]
            set_cell_background(c, "1F4E79")
            r = c.paragraphs[0].add_run(h_text)
            r.font.bold = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(255, 255, 255)

        for idx, m_key in enumerate(avail, start=1):
            m = all_cfg[m_key]
            vals = [
                m["model_name"],
                m["architecture"],
                m["adaptation_method"],
                f"{m['bleu']:.2f}",
                f"{m['chrf']:.2f}",
                f"{m['train_time_sec']:.1f}s",
                f"{m['latency_ms_per_sent']:.1f} ms",
            ]
            for j, v_text in enumerate(vals):
                c = res_table.rows[idx].cells[j]
                if idx % 2 == 0:
                    set_cell_background(c, "F9F9F9")
                r = c.paragraphs[0].add_run(v_text)
                r.font.size = Pt(9.0)

    # Section 3: Key Findings & Responsible AI Summary
    h3 = doc.add_heading("3. Key Architectural, Temperature & Ethical AI Findings", level=2)
    h3.style.font.color.rgb = RGBColor(31, 78, 121)

    p_f = doc.add_paragraph()
    p_f.add_run("• Encoder-Decoder vs. Decoder-Only Comparison: ").bold = True
    p_f.add_run(
        "The from-scratch Basic Transformer achieves fast inference latency with a compact parameter footprint (6.27M parameters), "
        "while pretrained multilingual models (mT5-small, LLaMA-3.2-1B, and Gemma-3-1B) leverage broad subword vocabularies and multilingual "
        "pretraining to handle rare words, complex syntax, and idiomatic expressions with substantially higher BLEU and chrF scores.\n"
    )
    p_f.add_run("• Impact of Adaptation Method (Prompting vs. LoRA): ").bold = True
    p_f.add_run(
        "Applying parameter-efficient LoRA fine-tuning (updating <0.3% of parameters) to LLaMA-3.2-1B and Gemma-3-1B aligns their outputs "
        "to target domain conciseness and French punctuation conventions, significantly boosting BLEU and chrF over 3-shot prompting while "
        "eliminating multi-shot prompt overhead during inference.\n"
    )
    p_f.add_run("• Sampling-Temperature Analysis: ").bold = True
    p_f.add_run(
        "Greedy decoding (T = 0.0) and low temperature (T = 0.3) maximize translation fidelity (BLEU and chrF), whereas higher temperatures "
        "(T = 0.7, 1.0) increase lexical diversity (Distinct-2) at the cost of semantic drift and reduced n-gram precision.\n"
    )
    p_f.add_run("• Responsible & Ethical AI Audit: ").bold = True
    p_f.add_run(
        "Diagnostic testing reveals occupational gender bias when translating gender-neutral English roles into French (defaulting 'doctor' "
        "and 'engineer' to masculine 'le médecin'/'l'ingénieur' and 'nurse'/'secretary' to feminine 'l'infirmière'/'la secrétaire'), alongside "
        "literal idiom translation risks. Mitigation measures including gender-balanced counterfactual data augmentation, constrained terminology "
        "decoding, and dual-gender output templates are implemented and proposed.\n"
    )
    p_f.add_run("• Best-Performing Model: ").bold = True
    p_f.add_run(
        f"{best_model_name} achieved the highest translation performance (BLEU: {best_bleu:.2f}, chrF: {best_chrf:.2f}), "
        "combining rich multilingual pretraining with parameter-efficient LoRA adaptation."
    )

    doc.save(output_filename)
    print(f"Successfully generated Word submission document: {output_filename}")


def create_submission_pdf(
    student_name: str = "Kumar Kshitij",
    registration_num: str = "230953488",
    github_link: str = "https://github.com/Kshiku12/Gen-AI-Assignment",
    output_filename: str = "KumarKshitij_230953488_FGAI_Assignment3.pdf",
):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    summary = load_results_summary()
    primary = summary["primary_models"] if summary else {}
    best_model_name = "Gemma-3-1B"
    best_bleu = 0.0
    best_chrf = 0.0
    if primary:
        best_model_name = max(primary.keys(), key=lambda k: primary[k]["bleu"])
        best_bleu = primary[best_model_name]["bleu"]
        best_chrf = primary[best_model_name]["chrf"]

    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=46,
        leftMargin=46,
        topMargin=46,
        bottomMargin=46,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1f4e79"),
        alignment=1,
    )
    sub_style = ParagraphStyle(
        "SubStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#595959"),
        alignment=1,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1f4e79"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#262626"),
    )
    bold_label = ParagraphStyle(
        "BoldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#1f4e79"),
    )
    link_style = ParagraphStyle(
        "LinkStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#0563c1"),
    )
    th_style = ParagraphStyle(
        "THStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )
    td_style = ParagraphStyle(
        "TDStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#222222"),
    )

    story = []
    story.append(
        Paragraph("Foundations of Generative AI (ICT4403)<br/>Assignment 3 Submission Document", title_style)
    )
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "Neural Machine Translation System (English to French) using Basic Transformer, mT5, LLaMA, and Gemma",
            sub_style,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Submission Details", h2_style))
    table_data = [
        [Paragraph("Student Name", bold_label), Paragraph(student_name, body_style)],
        [Paragraph("Registration Number", bold_label), Paragraph(registration_num, body_style)],
        [
            Paragraph("Assignment Title", bold_label),
            Paragraph("Foundations of Generative AI (ICT4403) - Assignment 3", body_style),
        ],
        [
            Paragraph("Title of Developed System", bold_label),
            Paragraph(
                "Comparative English-to-French Neural Machine Translation System using Encoder-Decoder (Basic Transformer, mT5) and Decoder-Only (LLaMA, Gemma) Architectures",
                body_style,
            ),
        ],
        [
            Paragraph("Dataset Name and Link", bold_label),
            Paragraph(
                "Tatoeba / OPUS English-French Parallel Corpus<br/><u>https://www.manythings.org/anki/fra-eng.zip</u> | <u>https://huggingface.co/datasets/Helsinki-NLP/opus-100</u>",
                link_style,
            ),
        ],
        [Paragraph("GitHub Repository Link", bold_label), Paragraph(f"<u>{github_link}</u>", link_style)],
    ]
    t = Table(table_data, colWidths=[145, 375])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Executive Summary & Quantitative Evaluation", h2_style))
    story.append(
        Paragraph(
            "This project develops, adapts, and empirically evaluates an English-to-French (en -&gt; fr) Neural Machine Translation "
            "system across four distinct generative architectures: (1) a Basic Encoder-Decoder Transformer trained from scratch in PyTorch, "
            "(2) mT5-small (google/mt5-small, Encoder-Decoder) fine-tuned via supervised Seq2Seq learning, "
            "(3) LLaMA-3.2-1B-Instruct (Decoder-Only) adapted via both 3-Shot In-Context Prompting and LoRA Fine-Tuning, and "
            "(4) Gemma-3-1B-IT (Decoder-Only) adapted via both 3-Shot In-Context Prompting and LoRA Fine-Tuning. "
            "All models were evaluated on the identical held-out test set from the Tatoeba English-French Parallel Corpus "
            "(12,000 sentence pairs: 9,600 train, 1,200 validation, 1,200 test).",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    if summary and "all_configurations" in summary:
        all_cfg = summary["all_configurations"]
        order = [
            "Basic Transformer",
            "mT5-small",
            "LLaMA-3.2-1B (Prompting)",
            "LLaMA-3.2-1B",
            "Gemma-3-1B (Prompting)",
            "Gemma-3-1B",
        ]
        avail = [k for k in order if k in all_cfg]
        res_rows = [
            [
                Paragraph("Model", th_style),
                Paragraph("Architecture", th_style),
                Paragraph("Adaptation", th_style),
                Paragraph("BLEU", th_style),
                Paragraph("chrF", th_style),
                Paragraph("Train (s)", th_style),
                Paragraph("Latency", th_style),
            ]
        ]
        for m_key in avail:
            m = all_cfg[m_key]
            res_rows.append(
                [
                    Paragraph(m["model_name"], td_style),
                    Paragraph(m["architecture"], td_style),
                    Paragraph(m["adaptation_method"], td_style),
                    Paragraph(f"{m['bleu']:.2f}", td_style),
                    Paragraph(f"{m['chrf']:.2f}", td_style),
                    Paragraph(f"{m['train_time_sec']:.1f}s", td_style),
                    Paragraph(f"{m['latency_ms_per_sent']:.1f} ms", td_style),
                ]
            )
        rt = Table(res_rows, colWidths=[102, 92, 122, 42, 42, 52, 68])
        rt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(rt)
        story.append(Spacer(1, 8))

    story.append(Paragraph("3. Key Architectural, Temperature & Ethical AI Findings", h2_style))
    story.append(
        Paragraph(
            "<b>• Encoder-Decoder vs. Decoder-Only Comparison:</b> The from-scratch Basic Transformer achieves ultra-low inference latency with a compact parameter footprint (6.27M parameters), whereas pretrained multilingual models (mT5-small, LLaMA-3.2-1B, and Gemma-3-1B) leverage massive subword vocabularies and web-scale pretraining to handle rare words, complex syntax, and idioms with substantially higher BLEU and chrF scores.<br/><br/>"
            "<b>• Impact of Adaptation Method (Prompting vs. LoRA):</b> Applying parameter-efficient LoRA fine-tuning (updating &lt;0.3% of parameters) to LLaMA-3.2-1B and Gemma-3-1B aligns their outputs to target domain conciseness and French punctuation conventions, significantly boosting BLEU and chrF over 3-shot prompting while eliminating multi-shot prompt overhead during inference.<br/><br/>"
            "<b>• Sampling-Temperature Analysis:</b> Greedy decoding (T = 0.0) and low temperature (T = 0.3) maximize translation fidelity (BLEU and chrF), whereas higher temperatures (T = 0.7, 1.0) increase lexical diversity (Distinct-2) at the cost of semantic drift.<br/><br/>"
            "<b>• Responsible &amp; Ethical AI Audit:</b> Diagnostic testing reveals occupational gender bias when translating gender-neutral English roles into French (defaulting 'doctor' and 'engineer' to masculine 'le médecin'/'l'ingénieur' and 'nurse'/'secretary' to feminine 'l'infirmière'/'la secrétaire'). Mitigation measures including counterfactual gender data augmentation, terminology-constrained decoding, and dual-gender templates are proposed.<br/><br/>"
            f"<b>• Best-Performing Model:</b> {best_model_name} is identified as the best-performing translation system (BLEU: {best_bleu:.2f}, chrF: {best_chrf:.2f}).",
            body_style,
        )
    )

    doc.build(story)
    print(f"Successfully generated PDF submission document: {output_filename}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Kumar Kshitij"
    reg = sys.argv[2] if len(sys.argv) > 2 else "230953488"
    gh = sys.argv[3] if len(sys.argv) > 3 else "https://github.com/Kshiku12/Gen-AI-Assignment"
    create_submission_word_doc(name, reg, gh)
    create_submission_pdf(name, reg, gh)

