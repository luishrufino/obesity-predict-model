Write-Host "Treinando o modelo..."
docker-compose run trainer

Write-Host "Iniciando a API..."
docker-compose up -d api

Write-Host "Verificando API..."
while ($true) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/" -UseBasicParsing -TimeoutSec 2
        Write-Host "API está ativa!"
        break
    } catch {
        Write-Host "Aguardando API..."
        Start-Sleep -Seconds 2
    }
}

Write-Host "Iniciando Streamlit..."
docker-compose up -d streamlit

Write-Host "✅ Ambiente iniciado com sucesso!"
