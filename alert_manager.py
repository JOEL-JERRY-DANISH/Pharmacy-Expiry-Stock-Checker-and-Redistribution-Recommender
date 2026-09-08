# alert_manager.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Replace with your Gmail address and app password
EMAIL_SENDER   = "yourpharmacy@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "pharmacist@pharmacy.com"

def send_alert(critical_items):
    """Send email alert for critical near-expiry items."""
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
    body += "http://localhost:8501\n"

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