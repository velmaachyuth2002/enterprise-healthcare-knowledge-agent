import enum

from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base
from app.models._enum_utils import enum_values


class UserRole(str, enum.Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    hashed_password: Mapped[str]
    role: Mapped[UserRole] = mapped_column(SqlEnum(UserRole, values_callable=enum_values))
