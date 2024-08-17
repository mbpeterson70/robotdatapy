class NoDataNearTimeException(Exception):
    
    def __init__(self, t_desired, t_closest=None):
        self.t_desired = t_desired
        self.t_closest = t_closest
        message = f"Desired time: {t_desired}. Closest time: {t_closest}"
        super().__init__(message)

class MsgNotFound(Exception):
    
    def __init__(self, topic, path=None):
        message = f"Message from topic {topic} not found"
        if path is not None:
            message += f" in path {path}"
        super().__init__(message)