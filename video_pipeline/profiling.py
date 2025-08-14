# import statements
import time
from functools import wraps

# main benchmarking decorator
def benchmark(func):
    """
    Decorator to benchmark a function's runtime.
    Prints and returns the runtime in seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Starts the start and end timer then uses the difference as the runtime
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        runtime = end - start
        print(f"[BENCHMARK] {func.__name__} took {runtime:.2f} seconds")
        return result, runtime
    return wrapper
