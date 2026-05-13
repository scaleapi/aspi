"""Project-wide constants.

Kept in a single module so both dataset construction and evaluation
read the same values. Historically BENCH_VERSION was defined in two
files; now they both import from here.
"""

# AgentDojo suite version we target. This is the benchmark version
# passed to `get_suite(BENCH_VERSION, ...)` everywhere.
BENCH_VERSION = "v1.2.2"

# Canonical list of AgentDojo suites we evaluate.
KNOWN_SUITES = ["workspace", "slack", "travel", "banking"]

# Canonical set of injection operators for C4_user variants. Stage 2
# always runs all three per pair.
STANDARD_OPERATORS = ["HR", "GS", "CP"]

# Default delay between LLM calls (per-thread, in seconds). Individual
# scripts can override via CLI.
DEFAULT_API_DELAY = 0.5

# Default worker count for parallel ThreadPoolExecutor fan-out.
DEFAULT_WORKERS = 4

# Default max retry attempts on QC failure during dataset construction.
MAX_RETRIES = 2
