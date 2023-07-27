# imports
import subprocess
from pathlib import Path
import pytest


TEST_ROODIR = Path(__file__).parent
EXEC_ROOTDIR = Path(__file__).parent.parent

def exec_command(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    """
    Executes shell command returns stdout if completes exit code 0

    :param cmnd: shell command to be executed
    :param stdout: stream
    :param sterr: stream
    """

    proc = subprocess.Popen(cmnd, shell=True, stdout=stdout, stderr=stderr)
    out, err = proc.communicate(

        if proc.returncode != 0:
            raise RuntimeError(f"FAILED: {cmnd}\n{err}")
    return out.decode("utf8") if out is not None else None


@pytest.fixture(scope='session')
def test_phynteny_command():
    """
    Test phynteny annotation command
    """
    cmd = f"phynteny test"
    exec_command(cmd)

# add tests for the training generation and confidence computation