#!/bin/bash
# MAPA_FRENTES - Script para actualizar GitHub
# Uso: bash push.sh "mensaje del commit"

cd "$(dirname "$0")"

if [ -z "$1" ]; then
  echo "Uso: bash push.sh \"mensaje del commit\""
  exit 1
fi

echo "=== Estado actual ==="
git status --short

echo ""
echo "=== Añadiendo cambios ==="
git add -A

echo ""
echo "=== Commit ==="
git commit -m "$1"

echo ""
echo "=== Push a GitHub ==="
git push origin main

echo ""
echo "=== Hecho ==="
git log --oneline -3
