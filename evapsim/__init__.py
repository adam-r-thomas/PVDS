
from evapsim import simulator
import logging

if __name__ == '__main__':
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
    sim = simulator.Simulator()
