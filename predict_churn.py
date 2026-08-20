"""predict_churn.py

CLI wrapper for production batch inference.
"""

from __future__ import annotations

import sys
from src.inference.predict_churn import main

if __name__ == "__main__":
    main()
