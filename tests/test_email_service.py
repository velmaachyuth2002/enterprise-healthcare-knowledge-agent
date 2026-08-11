import pytest
from aiosmtpd.controller import Controller

from app.services.email_service import EmailService

# A fixed port distinct from the documented local dev catcher (2525), so
# running the test suite never collides with a developer's own catcher
# already running for manual testing.
_TEST_SMTP_PORT = 10251


class _RecordingHandler:
    def __init__(self) -> None:
        self.received: list[dict] = []

    async def handle_DATA(self, server, session, envelope):
        self.received.append(
            {
                "from": envelope.mail_from,
                "to": envelope.rcpt_tos,
                "content": envelope.content.decode("utf-8", errors="replace"),
            }
        )
        return "250 Message accepted for delivery"


@pytest.fixture
def smtp_catcher():
    # A real, throwaway, in-process SMTP server - not a mocked smtplib
    # call. Proves EmailService actually speaks SMTP correctly, the same
    # way the SQL tools are tested against a real SQLite session rather
    # than a faked-out one.
    handler = _RecordingHandler()
    controller = Controller(handler, hostname="localhost", port=_TEST_SMTP_PORT)
    controller.start()
    try:
        yield handler
    finally:
        controller.stop()


def test_send_delivers_the_email_over_real_smtp(smtp_catcher) -> None:
    service = EmailService(
        host="localhost", port=_TEST_SMTP_PORT, from_address="noreply@medflow.example"
    )

    service.send(
        to="manager@medflow.example",
        subject="Escalation needs review",
        body="Ticket #42 requested for URGENT priority.",
    )

    assert len(smtp_catcher.received) == 1
    message = smtp_catcher.received[0]
    assert message["from"] == "noreply@medflow.example"
    assert message["to"] == ["manager@medflow.example"]
    assert "Escalation needs review" in message["content"]
    assert "Ticket #42 requested for URGENT priority." in message["content"]


def test_send_raises_if_no_smtp_server_is_reachable() -> None:
    # No catcher started for this test - confirms failures surface loudly
    # rather than being silently swallowed.
    service = EmailService(
        host="localhost", port=_TEST_SMTP_PORT, from_address="noreply@medflow.example"
    )

    with pytest.raises(OSError):
        service.send(to="manager@medflow.example", subject="x", body="x")
