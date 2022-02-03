
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

import logging


logging.basicConfig(filename='Evap_Sim.log',
                    encoding='utf-8',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filemode='w')
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
log.addHandler(stream)

log.info("Program Start")
sim = Simulator()
log.info("Simulator running:")

window = QtWidgets.QApplication(sys.argv)
app = Application_Qt(sim)
log.info("Application Window running:")
app.MainWindow.show()

sim.simulation_parameters()
sys.exit(window.exec_())
