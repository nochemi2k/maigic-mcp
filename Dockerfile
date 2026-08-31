FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=${MCP_PORT}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp/ /app/mcp/
COPY maigic/ /app/maigic/

EXPOSE ${MCP_PORT}

CMD ["python", "mcp/server.py"]
