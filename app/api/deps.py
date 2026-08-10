from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.session import get_db
from app.models.user import User, UserRole
from app.services.auth import InvalidTokenError, decode_access_token

# tokenUrl points Swagger's "Authorize" button at /login - it doesn't affect
# request handling, only the interactive docs UI.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

_INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    try:
        user_id = decode_access_token(
            token, secret=settings.jwt_secret, algorithm=settings.jwt_algorithm
        )
    except InvalidTokenError as exc:
        raise _INVALID_CREDENTIALS from exc

    # Re-fetched from the DB on every request rather than trusting a role
    # embedded in the token, so a role change or deleted account takes
    # effect immediately instead of only after the token expires.
    user = session.get(User, user_id)
    if user is None:
        raise _INVALID_CREDENTIALS
    return user


def require_role(role: UserRole):
    """Dependency factory: `Depends(require_role(UserRole.MANAGER))`.

    401 (via get_current_user) and 403 here are deliberately different
    failure modes - "not logged in" isn't the same problem as "logged in
    but not allowed to do this," and callers need to be able to tell them
    apart.
    """

    def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )
        return current_user

    return _check
