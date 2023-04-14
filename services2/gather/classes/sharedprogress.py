class SharedProgress:
    """
    This class is used to track the progress of a download task.
    """

    # Singleton instance
    _instance = None
    _progress = 0
    _total = 0
    _status = 'stopped'
    _message = 'Download started'
    _error = ''

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SharedProgress, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        pass

    def get_progress(self):
        return self._progress

    def get_total(self):
        return self._total

    def get_status(self):
        return self._status

    def get_message(self):
        return self._message

    def get_error(self):
        return self._error

    def set_progress(self, progress):
        self._progress = progress
        return self

    def set_total(self, total):
        self._total = total
        return self

    def set_status(self, status):
        self._status = status
        return self

    def set_message(self, message):
        self._message = message
        return self

    def set_error(self, error):
        self._error = error
        return self

    def to_dict(self):
        return {
            'progress': self._progress,
            'total': self._total,
            'status': self._status,
            'message': self._message,
            'error': self._error
        }

    def reset(self):
        self._progress = 0
        self._total = 0
        self._status = 'in progress'
        self._message = 'Download started'
        self._error = ''
        return self

    def __str__(self):
        return str(self.to_dict())