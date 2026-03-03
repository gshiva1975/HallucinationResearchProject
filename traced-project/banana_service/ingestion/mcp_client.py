# banana_service/ingestion/mcp_client.py
import time
import uuid
import requests
from logger import setup_logger

logger = setup_logger("MCPClient")


class MCPClient:

    def __init__(self, url: str, timeout: int = 5):
        self.url     = url
        self.timeout = timeout
        logger.info(f"MCPClient initialised  url={url}  timeout={timeout}s")

    def call_tool(self, tool_name: str, arguments: dict | None = None) -> list:
        request_id = str(uuid.uuid4())[:8]
        payload    = {
            "jsonrpc": "2.0",
            "method":  "tools/call",
            "params":  {"name": tool_name, "arguments": arguments or {}},
            "id":      request_id,
        }
        logger.info(f"→ MCP call  id={request_id}  url={self.url}  "
                    f"tool={tool_name}  args={arguments}")
        t0 = time.perf_counter()
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            elapsed  = round(time.perf_counter() - t0, 3)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"← MCP error  id={request_id}  elapsed={elapsed}s  "
                             f"error={data['error']}")
                raise RuntimeError(f"MCP Error: {data['error']}")

            result = data.get("result", [])
            logger.info(f"← MCP ok    id={request_id}  elapsed={elapsed}s  "
                        f"items={len(result) if isinstance(result, list) else 1}")
            return result

        except requests.exceptions.RequestException as e:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.error(f"← MCP fail  id={request_id}  elapsed={elapsed}s  error={e}")
            raise RuntimeError(f"MCP request failed: {e}")
