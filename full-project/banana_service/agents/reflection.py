
from logger import setup_logger
logger = setup_logger("Reflection")

class ReflectionAgent:
    def run(self, state):
        conf = state["sentiment"]["confidence"]
        threshold = state["threshold"]
        logger.info(f"Confidence={conf} | Threshold={threshold}")
        if conf >= threshold:
            state["proceed"] = True
        else:
            state["proceed"] = False
            state["report"] = "⚠ Low confidence. Autonomous stop."
        return state
