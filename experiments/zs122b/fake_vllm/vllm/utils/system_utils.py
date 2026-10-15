# 실제 vLLM 0.30.0 vllm/utils/system_utils.py의 suppress_stdout와 같은 동작 (시험용 사본)
import contextlib, os, sys

@contextlib.contextmanager
def suppress_stdout():
    if os.environ.get("VLLM_LOGGING_LEVEL") == "DEBUG":
        yield
        return
    stdout_fd = sys.stdout.fileno()
    stdout_dup = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(stdout_dup, stdout_fd)
        os.close(stdout_dup)
        os.close(devnull_fd)
