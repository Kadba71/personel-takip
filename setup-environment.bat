@echo off
echo 🚀 Personel Takip Projesi - Environment Setup
echo ================================================

echo.
echo 1️⃣ Python versiyonu kontrol ediliyor...
python --version
if errorlevel 1 (
    echo ❌ Python bulunamadı! Lütfen Python 3.11+ kurun.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo 2️⃣ Pip versiyonu kontrol ediliyor...
pip --version
if errorlevel 1 (
    echo ❌ Pip bulunamadı!
    pause
    exit /b 1
)

echo.

echo 3️⃣ Virtual environment oluşturuluyor...
cd backend
python -m venv .venv
if errorlevel 1 (
    echo ❌ Virtual environment oluşturulamadı!
    pause
    exit /b 1
)

echo.
echo 4️⃣ Virtual environment aktifleştiriliyor...
call .venv\Scripts\activate.bat

echo.
echo 5️⃣ Python paketleri yükleniyor...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Paket yükleme başarısız!
    pause
    exit /b 1
)

echo.
echo 6️⃣ Node.js kontrol ediliyor...
cd ..\frontend-react
node --version
if errorlevel 1 (
    echo ⚠️ Node.js bulunamadı! React frontend için Node.js gerekli.
    echo https://nodejs.org/
) else (
    echo 7️⃣ React dependencies yükleniyor...
    npm install
    if errorlevel 1 (
        echo ⚠️ NPM package yükleme başarısız!
    )
)

echo.
echo ✅ Environment setup tamamlandı!
echo.
echo 📋 Projeyi çalıştırmak için:
echo    Backend: cd backend && .venv\Scripts\activate && python main.py
echo    Frontend: cd frontend-react && npm run dev
echo.
pause
