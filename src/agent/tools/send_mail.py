import os
import smtplib
import ssl
from email.message import EmailMessage


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def send_email(receiver_email: str, subject: str, body: str) -> None:
    sender_email = os.getenv("EMAIL_USER")
    app_password = os.getenv("EMAIL_APP_PASSWORD")
    print(f"Attempting to send email from {sender_email} to {receiver_email} with subject '{subject}'")

    if not sender_email or not app_password:
        raise ValueError(
            "Missing EMAIL_USER or EMAIL_APP_PASSWORD environment variable"
        )

    message = EmailMessage()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.set_content(body)

    ssl_context = ssl.create_default_context()

    with smtplib.SMTP_SSL(
        SMTP_HOST,
        SMTP_PORT,
        context=ssl_context,
        timeout=15,
    ) as smtp:
        smtp.login(sender_email, app_password)
        smtp.send_message(message)