import smtplib
import mimetypes
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

# Example credentials for local testing. Replace these with real values when you
# are ready to send actual mail.
SMTP_USER = "use"
SMTP_PASS = "pass"
SMTP_SENDER = "use"

def send_status_email(image_path, now_date, to_list):
    """Sends the status email using the module-level SMTP credentials."""

    msg = EmailMessage()
    msg["Subject"] = f"ScintPi Status Update {now_date:%Y-%m-%d}"
    msg["From"] = SMTP_SENDER
    msg["To"] = ", ".join(to_list)
    
    msg.set_content("Attached: ScintPi availability summary.")
    cid = make_msgid(domain="scintpi")
    
    msg.add_alternative(f"""
    <html><body>
    <p>Attached is the ScintPi availability plot. Summaries are based on raw level files for {now_date:%Y-%m-%d}.</p>
    <img src="cid:{cid[1:-1]}" alt="Availability" />
    </body></html>""", subtype="html")

    with open(image_path, "rb") as f:
        img_bytes = f.read()
        
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    maintype, subtype = mime.split("/")
    msg.get_payload()[1].add_related(img_bytes, maintype=maintype, subtype=subtype, cid=cid)
    msg.add_attachment(img_bytes, maintype=maintype, subtype=subtype, filename=Path(image_path).name)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")