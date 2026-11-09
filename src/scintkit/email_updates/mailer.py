"""Send the availability plot by email over SMTP."""

import mimetypes
import os
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path


def send_status_email(image_path, now_date, to_list):
    """Email the plot inline and as an attachment.

    Reads SMTP_USER, SMTP_PASS, SMTP_SENDER (defaults to SMTP_USER) and
    optionally SMTP_HOST / SMTP_PORT from the environment. Errors propagate.
    """
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    if not user or not password:
        raise RuntimeError("SMTP_USER and SMTP_PASS must be set in the environment")
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))

    msg = EmailMessage()
    msg["Subject"] = f"ScintPi Status Update {now_date:%Y-%m-%d}"
    msg["From"] = os.environ.get("SMTP_SENDER", user)
    msg["To"] = ", ".join(to_list)
    msg.set_content("Attached: ScintPi availability summary.")
    cid = make_msgid(domain="scintpi")
    msg.add_alternative(f"""
    <html><body>
    <p>Attached is the ScintPi availability plot. Summaries are based on raw level files for {now_date:%Y-%m-%d}.</p>
    <img src="cid:{cid[1:-1]}" alt="Availability" />
    </body></html>""", subtype="html")

    image_bytes = Path(image_path).read_bytes()
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    maintype, subtype = mime.split("/")
    msg.get_payload()[1].add_related(image_bytes, maintype=maintype, subtype=subtype, cid=cid)
    msg.add_attachment(image_bytes, maintype=maintype, subtype=subtype, filename=Path(image_path).name)

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
