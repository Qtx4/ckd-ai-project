import yagmail

EMAIL = "kkabdwal178@gmail.com"
APP_PASSWORD = "ykob kmar acwp gkjd"

def send_report(receiver_email, pdf_path):
    try:
        yag = yagmail.SMTP(EMAIL, APP_PASSWORD)

        yag.send(
            to=receiver_email,
            subject="CKD Medical Report",
            contents="""
Hello,

Your CKD Report is attached.

Thank you for using CKD Stage Prediction AI.
""",
            attachments=pdf_path
        )

        return True

    except Exception as e:
        print(e)
        return False