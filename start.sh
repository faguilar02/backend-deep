#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} \
  --workers 1 \
  --limit-max-requests 50 \
  --timeout-keep-alive 10