import enum


def enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    """Use for `values_callable=` on every SQLAlchemy `Enum` column.

    By default SQLAlchemy's Enum type persists the member *name* ("OPEN"),
    not its value ("open"), unless told otherwise. Forcing it to use
    `.value` keeps what's stored in the database consistent with the
    Python string the rest of the app (and generated SQL) compares
    against.
    """
    return [member.value for member in enum_cls]
