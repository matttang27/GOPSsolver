"""
GOPS (Game of Pure Strategy) Solver
Main entry point for running solver calculations.
"""

import sys
from tests.actions import runFull, profile, findGuaranteeThreshold

def main():
    """Main entry point with command line options"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--profile":
            profile()
        elif sys.argv[1] == "--search":
           findGuaranteeThreshold()
        elif sys.argv[1] == "--help":
            print("GOPS Solver Options:")
            print("  (no args)    - Run default calculations")
            print("  --profile    - Run with profiling")
            print("  --search     - Binary search for guarantees")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            sys.exit(1)
    else:
        runFull()

if __name__ == "__main__":
    main()