import re
import json
from fpdf import FPDF
import os

def parse_feedback(feedback_text):
    def extract_score(text, label):
        match = re.search(rf"{label}:\s*(\d+)", text)
        return int(match.group(1)) if match else None

    def extract_list(text, label):
        section = re.search(rf"{label}:(.*?)(\n[A-Z][a-z]+:|\Z)", text, re.DOTALL)
        if section:
            items = section.group(1).strip().split('\n')
            return [item.strip('-• ') for item in items if item.strip()]
        return []

    def extract_category_scores(text):
        categories = {}
        cat_section = re.search(r"Category Scores:(.*?)(\n[A-Z][a-z]+:|\Z)", text, re.DOTALL)
        if cat_section:
            for line in cat_section.group(1).strip().split('\n'):
                parts = line.strip().split(':')
                if len(parts) == 2:
                    cat, score = parts[0].strip(), parts[1].strip()
                    if score.isdigit():
                        categories[cat] = int(score)
        return categories

    return {
        "overall_score": extract_score(feedback_text, "Overall Score"),
        "category_scores": extract_category_scores(feedback_text),
        "strengths": extract_list(feedback_text, "Strengths"),
        "areas_for_improvement": extract_list(feedback_text, "Areas for Improvement"),
        "recommendations": extract_list(feedback_text, "Recommendations"),
        "suggested_resources": extract_list(feedback_text, "Suggested Resources")
    }

def generate_pdf_from_json(data, filename="interview_feedback.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    def write_section(title, content):
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", size=12)
        if isinstance(content, list):
            for item in content:
                pdf.cell(0, 10, f"- {item}", ln=True)
        elif isinstance(content, dict):
            for key, value in content.items():
                pdf.cell(0, 10, f"{key}: {value}", ln=True)
        else:
            pdf.cell(0, 10, str(content), ln=True)
        pdf.ln(5)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Interview Feedback Summary", ln=True)
    pdf.ln(10)

    write_section("Overall Score", data.get("overall_score"))
    write_section("Category Scores", data.get("category_scores"))
    write_section("Strengths", data.get("strengths"))
    write_section("Areas for Improvement", data.get("areas_for_improvement"))
    write_section("Recommendations", data.get("recommendations"))
    write_section("Suggested Resources", data.get("suggested_resources"))

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)
    pdf.output(pdf_path)
    return pdf_path

def generate_feedback_outputs(feedback_text):
    structured_data = parse_feedback(feedback_text)
    pdf_path = generate_pdf_from_json(structured_data)
    return {
        "json": structured_data,
        "pdf_path": pdf_path
    }
