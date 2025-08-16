# 🚀 Personel Takip Projesi - Environment Setup
# ================================================

Write-Host "🚀 Personel Takip Projesi - Environment Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# 1. Python kontrol
Write-Host "`n1️⃣ Python versiyonu kontrol ediliyor..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python bulundu: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python bulunamadı"
    }
} catch {
    Write-Host "❌ Python bulunamadı! Lütfen Python 3.11+ kurun." -ForegroundColor Red
    Write-Host "https://www.python.org/downloads/" -ForegroundColor Blue
    Read-Host "Devam etmek için Enter'a basın"
    exit 1
}

# 2. Pip kontrol
Write-Host "`n2️⃣ Pip versiyonu kontrol ediliyor..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Pip bulundu: $pipVersion" -ForegroundColor Green
    } else {
        throw "Pip bulunamadı"
    }
} catch {
    Write-Host "❌ Pip bulunamadı!" -ForegroundColor Red
    Read-Host "Devam etmek için Enter'a basın"
    exit 1
}

# 3. Backend dizinine geç
Write-Host "`n3️⃣ Backend dizinine geçiliyor..." -ForegroundColor Yellow
Set-Location -Path "backend" -ErrorAction Stop

# 4. Virtual environment oluştur
Write-Host "`n4️⃣ Virtual environment oluşturuluyor..." -ForegroundColor Yellow
try {
    python -m venv venv
    Write-Host "✅ Virtual environment oluşturuldu" -ForegroundColor Green
} catch {
    Write-Host "❌ Virtual environment oluşturulamadı!" -ForegroundColor Red
    Read-Host "Devam etmek için Enter'a basın"
    exit 1
}

# 5. Virtual environment aktifleştir
Write-Host "`n5️⃣ Virtual environment aktifleştiriliyor..." -ForegroundColor Yellow
try {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment aktifleştirildi" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Virtual environment aktifleştirme hatası. Manuel olarak aktifleştirin." -ForegroundColor Yellow
}

# 6. Python paketleri yükle
Write-Host "`n6️⃣ Python paketleri yükleniyor..." -ForegroundColor Yellow
try {
    pip install --upgrade pip
    pip install -r requirements.txt
    Write-Host "✅ Python paketleri yüklendi" -ForegroundColor Green
} catch {
    Write-Host "❌ Paket yükleme başarısız!" -ForegroundColor Red
    Read-Host "Devam etmek için Enter'a basın"
}

# 7. Node.js kontrol
Write-Host "`n7️⃣ Node.js kontrol ediliyor..." -ForegroundColor Yellow
Set-Location -Path "..\frontend-react"
try {
    $nodeVersion = node --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Node.js bulundu: $nodeVersion" -ForegroundColor Green
        
        # NPM paketleri yükle
        Write-Host "`n8️⃣ React dependencies yükleniyor..." -ForegroundColor Yellow
        npm install
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ NPM paketleri yüklendi" -ForegroundColor Green
        } else {
            Write-Host "⚠️ NPM package yükleme başarısız!" -ForegroundColor Yellow
        }
    } else {
        throw "Node.js bulunamadı"
    }
} catch {
    Write-Host "⚠️ Node.js bulunamadı! React frontend için Node.js gerekli." -ForegroundColor Yellow
    Write-Host "https://nodejs.org/" -ForegroundColor Blue
}

# Tamamlama mesajı
Write-Host "`n✅ Environment setup tamamlandı!" -ForegroundColor Green
Write-Host "`n📋 Projeyi çalıştırmak için:" -ForegroundColor Cyan
Write-Host "   Backend: cd backend; .\venv\Scripts\Activate.ps1; python main.py" -ForegroundColor White
Write-Host "   Frontend: cd frontend-react; npm run dev" -ForegroundColor White
Write-Host ""

Read-Host "Devam etmek için Enter'a basın"
