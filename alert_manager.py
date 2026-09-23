# alert_manager.py
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EMAIL_SENDER   = os.getenv("EMAIL_SENDER",   "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
#: Public URL shown in alert emails.  Set APP_URL in .env to your deployment address.
APP_URL        = os.getenv("APP_URL",        "http://localhost:8501")

def send_alert(critical_items: list) -> bool:
    """Send an email alert for critical near-expiry items.

    Args:
        critical_items: List of recommendation dicts whose urgency is 'critical'.

    Returns:
        True if the email was delivered successfully, False otherwise.
    """
    if not critical_items:
        return False

    subject = (f"URGENT: {len(critical_items)} critical "
               f"medicines need action today")

    body = "The following medicines expire within 7 days:\n\n"
    for item in critical_items:
        body += (
            f"- {item['medicine_name']} | "
            f"{item['quantity']} units | "
            f"Expires in {item['dte']} days | "
            f"Worth £{item['stock_value']:.2f}\n"
            f"  Branch: {item['branch_name']}\n"
            f"  Recommendation: {item['reason']}\n\n"
        )

    body += "\nPlease log in to take action:\n"
    body += f"{APP_URL}\n"

    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False
