"""
Main entry point for the IOT distributed constraint optimization project.
"""

import logging

from iot.experiment import ExperimentRunner

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d – %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Run the experiment with default settings"""
    logger.info("Starting IOT dcop experiment")
    runner = ExperimentRunner(rounds=50)
    runner.run()
    logger.info("completed successfully")

if __name__ == "__main__":
   # main()

#####TRYING DSA####
    from iot.problems import PRESET_PROBLEMS
    from iot.agents import DSA
    runner = ExperimentRunner(rounds=50)
    runner._run_single(seed=1, prob=PRESET_PROBLEMS[1], algo_cls=DSA)
