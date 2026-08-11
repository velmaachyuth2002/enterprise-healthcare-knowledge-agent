import smtplib
from email.message import EmailMessage


class EmailService:
    """The single chokepoint every outbound email goes through - mirrors
    LlmGateway's role for LLM calls. Talks to a local dev SMTP catcher via
    smtplib for now (see README for how to run one); pointing this at a
    real provider later is a config change behind this same send() call,
    not a new call site."""

    def __init__(self, host: str, port: int, from_address: str) -> None:
        self._host = host
        self._port = port
        self._from_address = from_address

    def send(self, to: str, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self._from_address
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)

        with smtplib.SMTP(self._host, self._port) as smtp:
            smtp.send_message(message)
