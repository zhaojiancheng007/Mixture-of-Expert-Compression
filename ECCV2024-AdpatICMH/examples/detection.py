# detection.py
# Entry script for detection-oriented task-aware compression training.

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from task_coco_train import main_task


if __name__ == "__main__":
    main_task(
        sys.argv[1:],
        task_type="detection",
        default_config="config/detection.yaml",
    )
