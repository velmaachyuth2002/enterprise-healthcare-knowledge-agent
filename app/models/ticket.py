import enum
from datetime import datetime

from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


def _values(enum_cls: type[enum.Enum]) -> list[str]:
    # By default SQLAlchemy's Enum type persists the member *name*
    # ("OPEN"), not its value ("open"). Forcing it to use `.value` keeps
    # what's stored in the database consistent with the Python string the
    # rest of the app (and, later, generated SQL) actually compares against.
    return [member.value for member in enum_cls]


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str]
    status: Mapped[TicketStatus] = mapped_column(SqlEnum(TicketStatus, values_callable=_values))
    priority: Mapped[TicketPriority] = mapped_column(
        SqlEnum(TicketPriority, values_callable=_values)
    )
    created_at: Mapped[datetime]
    resolved_at: Mapped[datetime | None] = mapped_column(default=None)
