import logging
import sys
from functools import wraps

from PyQt5.QtWidgets import QApplication

_logger = logging.getLogger('ibllib')


def create_app():
    """Create a Qt application."""
    global QT_APP
    QT_APP = QApplication.instance()
    if QT_APP is None:  # pragma: no cover
        QT_APP = QApplication(sys.argv)
    return QT_APP


def require_qt(func):
    """Function decorator to specify that a function requires a Qt application.
    Use this decorator to specify that a function needs a running
    Qt application before it can run. An error is raised if that is not
    the case.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not QApplication.instance():  # pragma: no cover
            _logger.warning("Creating a Qt application.")
            create_app()
        return func(*args, **kwargs)
    return wrapped


@require_qt
def run_app():  # pragma: no cover
    """Run the Qt application."""
    global QT_APP
    return QT_APP.exit(QT_APP.exec_())
