
from logger import setup_logger
logger = setup_logger("Scribe")

class ScribeAgent:
    def run(self, state):
        logger.info("Generating final report")
        report = f"""
Financial Report:

Sentiment: {state['sentiment']['label']}
Confidence: {state['sentiment']['confidence']}
"""
        return {**state, "report": report}
