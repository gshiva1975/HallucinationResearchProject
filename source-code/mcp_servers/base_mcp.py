# mcp_servers/base_mcp.py
import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BaseMCP")


class JSONRPCRequest(BaseModel):
    jsonrpc: str
    method:  str
    params:  Optional[Dict[str, Any]] = {}
    id:      Optional[str | int] = None


class BaseMCP:

    def __init__(self):
        self.app   = FastAPI()
        self.tools = {}
        self._register_routes()
        logger.info("BaseMCP initialised")

    def register(self, name: str, func):
        self.tools[name] = func
        logger.info(f"Tool registered: {name}")

    def _error(self, code: int, message: str, id_value):
        logger.warning(f"JSON-RPC error  code={code}  message={message}")
        return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": id_value}

    def _success(self, result: Any, id_value):
        return {"jsonrpc": "2.0", "result": result, "id": id_value}

    def _register_routes(self):
        @self.app.post("/mcp")
        def handle(req: JSONRPCRequest):
            t0 = time.perf_counter()
            logger.info(f"→ JSON-RPC  id={req.id}  method={req.method}  params={req.params}")

            if req.jsonrpc != "2.0":
                return self._error(-32600, "Invalid JSON-RPC version", req.id)
            if req.method != "tools/call":
                return self._error(-32601, "Method not found", req.id)
            if not req.params:
                return self._error(-32602, "Invalid params", req.id)

            tool_name = req.params.get("name")
            arguments = req.params.get("arguments", {})
            logger.info(f"  tool={tool_name}  args={arguments}")

            if tool_name not in self.tools:
                return self._error(-32601, f"Tool not found: {tool_name}", req.id)

            try:
                result  = self.tools[tool_name](**arguments)
                elapsed = round(time.perf_counter() - t0, 3)
                logger.info(f"← OK  id={req.id}  elapsed={elapsed}s  "
                            f"result_items={len(result) if isinstance(result, list) else 1}")
                return self._success(result, req.id)
            except Exception as e:
                elapsed = round(time.perf_counter() - t0, 3)
                logger.error(f"← ERROR  id={req.id}  elapsed={elapsed}s  error={e}")
                return self._error(-32603, f"Internal error: {e}", req.id)
