"""Creates sample tickets for manual testing and demos - so you have
real, varied data to run ticket-count queries, unresolved-ticket
listings, and escalations against instead of inserting one ticket by
hand every time. Run with:

    uv run python -m scripts.seed_tickets

Safe to re-run: skips entirely if any tickets already exist, so repeated
runs don't pile up duplicates.
"""

import random
from datetime import UTC, datetime, timedelta

from app.database.session import Base, SessionLocal, engine
from app.models.ticket import Ticket, TicketPriority, TicketStatus

_SUBJECT_TEMPLATES = [
    "Claims submission blocked for {provider}",
    "Patient scheduling conflict at {provider}",
    "Provider staff login failure at {provider}",
    "Duplicate patient record reported by {provider}",
    "Billing discrepancy flagged by {provider}",
    "Integration sync failure for {provider}",
    "Data export request from {provider}",
    "Onboarding delay for {provider}",
]

_PROVIDERS = [
    "MedCore Clinic",
    "Riverside Family Health",
    "Sunrise Pediatrics",
    "Lakeside Urgent Care",
    "Harborview Medical Group",
    "Cedar Valley Clinic",
]

# Weighted so most tickets are open/medium, same shape a real queue has -
# a handful of urgent/closed outliers, not a uniform spread.
_STATUS_WEIGHTS = [
    (TicketStatus.OPEN, 4),
    (TicketStatus.IN_PROGRESS, 3),
    (TicketStatus.RESOLVED, 2),
    (TicketStatus.CLOSED, 1),
]
_PRIORITY_WEIGHTS = [
    (TicketPriority.LOW, 3),
    (TicketPriority.MEDIUM, 4),
    (TicketPriority.HIGH, 2),
    (TicketPriority.URGENT, 1),
]


def _weighted_choice(weighted: list[tuple]):
    population, weights = zip(*weighted)
    return random.choices(population, weights=weights, k=1)[0]


def seed(count: int = 100) -> None:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        existing = session.query(Ticket).count()
        if existing > 0:
            print(f"skip: {existing} ticket(s) already exist")
            return

        now = datetime.now(UTC)
        for _ in range(count):
            subject = random.choice(_SUBJECT_TEMPLATES).format(
                provider=random.choice(_PROVIDERS)
            )
            status = _weighted_choice(_STATUS_WEIGHTS)
            priority = _weighted_choice(_PRIORITY_WEIGHTS)
            created_at = now - timedelta(
                days=random.randint(0, 120), hours=random.randint(0, 23)
            )
            resolved_at = None
            if status in (TicketStatus.RESOLVED, TicketStatus.CLOSED):
                resolved_at = created_at + timedelta(days=random.randint(0, 10))

            session.add(
                Ticket(
                    subject=subject,
                    status=status,
                    priority=priority,
                    created_at=created_at,
                    resolved_at=resolved_at,
                )
            )
        session.commit()
        print(f"created {count} tickets")
    finally:
        session.close()


if __name__ == "__main__":
    seed()
