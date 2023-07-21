from common.base_error import BaseError


class ClusterError(BaseError):
    # 20XXX user
    NO_DATA = 20000
    DATA_NOT_FOUND = 20001
    USER_NOT_LOGIN = 20002
    USER_NOT_FOUND = 20003

    def error(self):
        return {'error': self.errno, 'errmsg': self.message}
