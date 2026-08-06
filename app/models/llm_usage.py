import enum
from datetime import datetime

from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base
from app.models._enum_utils import enum_values


class LlmUsageStatus(str, enum.Enum):
    SUCCESS = "success"
    ERROR = "error"
    BLOCKED = "blocked"


class LlmUsage(Base):
    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    feature: Mapped[str]
    model: Mapped[str]
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]
    cost_usd: Mapped[float]
    latency_ms: Mapped[int]
    status: Mapped[LlmUsageStatus] = mapped_column(
        SqlEnum(LlmUsageStatus, values_callable=enum_values)
    )
    created_at: Mapped[datetime]
