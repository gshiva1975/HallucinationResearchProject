from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional


class JSONRPCRequest(BaseModel):
    jsonrpc: str
    method: str
    params: Optional[Dict[str, Any]] = {}
    id: Optional[str | int] = None


class BaseMCP:
    def __init__(self):
        self.app = FastAPI()
        self.tools = {}
        self._register_routes()

    def register(self, name, func):
        self.tools[name] = func

    def _error(self, code: int, message: str, id_value):
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": id_value
        }

    def _success(self, result: Any, id_value):
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": id_value
        }

    def _register_routes(self):

        @self.app.post("/mcp")
        def handle(req: JSONRPCRequest):

            if req.jsonrpc != "2.0":
                return self._error(-32600, "Invalid JSON-RPC version", req.id)

            # We standardize on tools/call
            if req.method != "tools/call":
                return self._error(-32601, "Method not found", req.id)

            if not req.params:
                return self._error(-32602, "Invalid params", req.id)

            tool_name = req.params.get("name")
            arguments = req.params.get("arguments", {})

            if tool_name not in self.tools:
                return self._error(-32601, "Tool not found", req.id)

            try:
                result = self.tools[tool_name](**arguments)
                return self._success(result, req.id)
            except Exception as e:
                return self._error(-32603, f"Internal error: {str(e)}", req.id)
