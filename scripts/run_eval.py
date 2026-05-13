#!/usr/bin/env python3
"""ASPI evaluation CLI (full agent runner).
"""
import logging, os
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

import litellm
litellm.suppress_debug_info = True   # silences the "Give Feedback / Get Help" banner

from aspi.eval.runner import main

if __name__ == "__main__":
    main()
