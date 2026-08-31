# MaIGIC MCP (HTTP)

Удалённый MCP-сервер MaIGIC. Те же тулы (`determine_hamiltonian`, `compute_property`, …), транспорт Streamable HTTP.

Скопируйте **эту папку** на сервер (не весь родительский репозиторий).

## Запуск

```bash
cp .env.example .env
```

В `.env` поставьте длинный `MCP_TOKEN` (не оставляйте пример):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

```bash
docker compose up -d --build
```

Сервер слушает `http://SERVER:8000/mcp`. Проверка: `curl http://SERVER:8000/health`.

Откройте порт `8000` в файрволе. Если спереди nginx/Caddy с TLS — в `mcp.json` используйте `https://…`.

## mcp.json

В Cursor: Settings → MCP, или файл `.cursor/mcp.json`. Подставьте IP/домен сервера и тот же токен, что в `.env`.

```json
{
  "mcpServers": {
    "maigic": {
      "url": "http://YOUR_SERVER:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

Перезагрузите MCP-серверы в Cursor. После этого тулы MaIGIC доступны с любой машины, где прописан этот JSON.

## Остановка

```bash
docker compose down
```
