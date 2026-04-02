FROM python:3.12-slim

WORKDIR /app

# Copy entire repo so both banana_service/ and mcp_servers/ are importable
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "banana_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
