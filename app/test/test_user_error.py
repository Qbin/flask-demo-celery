from common.base_error import BaseError


class TestUserError(BaseError):
    # 20XXX user
    USER_AUTHENTICATED_EORROR = 20001
    USER_NOT_LOGIN = 20002
    USER_NOT_FOUND = 20003

    def error(self):
        return {'error': self.errno, 'errmsg': self.message}
