#!/usr/bin/env bash
# Build the scratchpad-memory-management benchmark.
#
# Pure analytical model of the paper (Section 3.2 footprint/accesses + Section 3.3
# Algorithm 1).  No LRU/cache: off-chip volume is closed-form, so there is NO
# cachemere / abseil / boost dependency anymore -- just the standard library.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$here/build"

g++ -std=c++17 -O2 -Wall -Wextra \
    -I"$here/include" \
    "$here/src/main.cpp" -o "$here/build/smm_bench"

# Explicit scratchpad-management mechanism: GLB partition map + tile schedule.
g++ -std=c++17 -O2 -Wall -Wextra \
    -I"$here/include" \
    "$here/src/schedule_demo.cpp" -o "$here/build/smm_schedule"

echo "built: $here/build/smm_bench"
echo "built: $here/build/smm_schedule"
