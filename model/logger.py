import logging
import time

formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

handler = logging.StreamHandler()
handler.setFormatter(formatter)

st = time.strftime("./log/%Y-%m-%d_%H%M_%S.log", time.localtime())
file_handler = logging.FileHandler(st)

logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)