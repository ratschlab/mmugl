# ===============================================
#
# Utility code around monitoring
#
# ===============================================
import subprocess
from typing import Dict


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.

    Source: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split("\n")]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map
    except:
        return {0: 0}
