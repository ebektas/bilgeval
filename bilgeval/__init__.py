__all__ = ["run_benchmark"]


def run_benchmark(*args, **kwargs):
    from .main import run_benchmark as implementation

    return implementation(*args, **kwargs)
