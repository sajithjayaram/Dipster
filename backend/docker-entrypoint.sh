#!/usr/bin/env sh
set -e

# Helper: export var from file if *_FILE is present or secret exists in /run/secrets
export_from_file() {
  VAR_NAME="$1"
  FILE_VAR_NAME="${VAR_NAME}_FILE"
  VAR_VALUE=$(printenv "$VAR_NAME")
  FILE_VALUE=$(printenv "$FILE_VAR_NAME")
  if [ -z "$VAR_VALUE" ]; then
    if [ -n "$FILE_VALUE" ] && [ -f "$FILE_VALUE" ]; then
      export "$VAR_NAME"="$(cat "$FILE_VALUE")"
    elif [ -f "/run/secrets/$VAR_NAME" ]; then
      export "$VAR_NAME"="$(cat "/run/secrets/$VAR_NAME")"
    fi
  fi
}

# Secrets we support
for key in OPENAI_API_KEY GOOGLE_CLIENT_SECRET TELEGRAM_BOT_TOKEN GOOGLE_CLIENT_ID; do
  export_from_file "$key"
done

# Default host/port
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}

exec uvicorn server:app --host "$HOST" --port "$PORT"