"""
    log_setup.py
    Matthew Flood
"""
import logging

def init(loglevel=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    channel = logging.StreamHandler()
    channel.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    channel.setFormatter(formatter)
    logger.addHandler(channel)

    return logger
