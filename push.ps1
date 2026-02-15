# MAPA_FRENTES - Script para actualizar GitHub
# Uso: .\push.ps1 "mensaje del commit"

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Mensaje
)

Set-Location $PSScriptRoot

Write-Host "=== Estado actual ===" -ForegroundColor Cyan
git status --short

Write-Host "`n=== Anadiendo cambios ===" -ForegroundColor Cyan
git add -A

Write-Host "`n=== Commit ===" -ForegroundColor Cyan
git commit -m $Mensaje

Write-Host "`n=== Push a GitHub ===" -ForegroundColor Cyan
git push origin main

Write-Host "`n=== Hecho ===" -ForegroundColor Green
git log --oneline -3
