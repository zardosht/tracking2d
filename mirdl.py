import sys
from PyQt5 import QtWidgets
import logging
from View.StartMenuWindow import StartMenuWindow


# create logger with 'spam_application'
logger = logging.getLogger('mirdl')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('mirdl.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Logging is configured.')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = StartMenuWindow()
    window.show()
    app.exec_()
