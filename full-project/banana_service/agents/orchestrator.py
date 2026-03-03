
from langgraph.graph import StateGraph
from logger import setup_logger

logger = setup_logger("Orchestrator")

class Orchestrator:
    def __init__(self, researcher, analyst, reflection, scribe, threshold):
        graph = StateGraph(dict)

        graph.add_node("research", researcher.run)
        graph.add_node("analysis", analyst.run)
        graph.add_node("reflection", reflection.run)
        graph.add_node("scribe", scribe.run)

        graph.set_entry_point("research")
        graph.add_edge("research","analysis")
        graph.add_edge("analysis","reflection")

        def decide(state):
            return "scribe" if state["proceed"] else "__end__"

        graph.add_conditional_edges("reflection", decide)

        self.workflow = graph.compile()
        self.threshold = threshold

    def run(self, query):
        logger.info("Starting agentic workflow")
        return self.workflow.invoke({
            "query": query,
            "threshold": self.threshold
        })
