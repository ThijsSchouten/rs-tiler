import time

def timer(logger):
    def decorator(function):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            duration = round(end-start, 2)
            logger.info(f"⏱️  /{function.__name__}/  {duration} seconds")
            return result
        return wrapper
    return decorator