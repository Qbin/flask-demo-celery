from common.base_error import BaseError


class ClusterError(BaseError):
    # 20XXX cluster
    SERVER_BUSY = 20000
    NO_DATA = 20001
    TEXT2VEC = 20002

    def error(self):
        return {'error': self.errno, 'errmsg': self.message}
