import logging
import os
def set_logger(save_path):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    file_handler = logging.FileHandler(f"{save_path}/train_log")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只输出 INFO 及以上级别的日志
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger