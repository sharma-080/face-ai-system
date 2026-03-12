#!/bin/bash
# Start FaceAI
cd "$(dirname "$0")"
uvicorn backend.main:app --host 0.0.0.0 --port 8000