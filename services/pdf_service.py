from fpdf import FPDF
import datetime

def generate_pdf(stage, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"CKD Report - {stage}", ln=True)
    pdf.multi_cell(0, 10, report_text)

    filename = f"report_{datetime.datetime.now().timestamp()}.pdf"
    pdf.output(filename)
    return filename