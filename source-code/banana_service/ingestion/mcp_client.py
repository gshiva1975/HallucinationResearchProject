# banana_service/ingestion/mcp_client.py

import requests
import uuid


class MCPClient:
    def __init__(self, url, timeout=5):
        self.url = url
        self.timeout = timeout

    def call_tool(self, tool_name: str, arguments: dict | None = None):
        request_id = str(uuid.uuid4())

        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            },
            "id": request_id
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # JSON-RPC error handling
            if "error" in data:
                raise RuntimeError(
                    f"MCP Error: {data['error']}"
                )

            return data.get("result", [])

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"MCP request failed: {e}")
