# Personel Takip Projesi - Sorun Giderme Rehberi

## 🔍 VS Code Yeniden Yükleme Sonrası Durum

### ❌ Tespit Edilen Sorunlar:

1. **Python PATH Problemi**
   - Python yüklü ama `python --version` çalışmıyor
   - `pip` komutu tanınmıyor
   - Python 3.7 görünüyor ama aktif değil

2. **Node.js Eksik**
   - React frontend için Node.js gerekli
   - `node --version` çalışmıyor

3. **Python Paketleri Eksik**
   - FastAPI, Uvicorn, Pandas vb. yüklü değil
   - Virtual environment yok

## 🛠️ Çözüm Adımları

### 1. Python Kurulumu (Gerekli)
```
1. https://www.python.org/downloads/ adresine gidin
2. Python 3.11+ versiyonunu indirin
3. Kurulum sırasında "Add Python to PATH" kutucuğunu işaretleyin
4. Kurulumu tamamlayın
5. PowerShell'i yeniden başlatın
```

### 2. Node.js Kurulumu (React için)
```
1. https://nodejs.org/ adresine gidin  
2. LTS versiyonunu indirin
3. Kurulumu tamamlayın
4. PowerShell'i yeniden başlatın
```

### 3. Otomatik Environment Setup
Python ve Node.js kurulduktan sonra:

**Windows PowerShell (Yönetici olarak):**
```powershell
cd "e:\DIŞ DATA\personel_takip.py"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup-environment.ps1
```

**Veya Command Prompt:**
```cmd
cd "e:\DIŞ DATA\personel_takip.py"
setup-environment.bat
```

### 4. Manuel Kurulum (Alternatif)
Otomatik script çalışmazsa:

```powershell
# Backend için
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Frontend için
cd ..\frontend-react
npm install
```

### 5. VS Code Extensions
Gerekli extension'ları yükleyin:
- Python
- Pylance  
- ES7+ React/Redux/React-Native snippets
- Auto Rename Tag
- GitLens

### 6. VS Code Python Interpreter
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. `backend\venv\Scripts\python.exe` seçin

## 🚀 Projeyi Çalıştırma

### Backend:
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python main.py
```

### Frontend (React):
```powershell
cd frontend-react  
npm run dev
```

### Frontend (Vanilla):
```powershell
cd backend
python main.py
# Tarayıcıda: http://localhost:8000
```

## ✅ Kontrol Listesi

- [ ] Python 3.11+ kuruldu ve PATH'te
- [ ] Node.js kuruldu ve PATH'te
- [ ] Virtual environment oluşturuldu
- [ ] Python paketleri yüklendi
- [ ] NPM paketleri yüklendi  
- [ ] VS Code Python interpreter seçildi
- [ ] Backend çalışıyor (port 8000)
- [ ] Frontend çalışıyor

## 🔧 VS Code Yapılandırması

### settings.json önerileri:
```json
{
    "python.pythonPath": "./backend/venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "typescript.updateImportsOnFileMove.enabled": "always",
    "editor.formatOnSave": true
}
```

### launch.json (Debugging):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python", 
            "request": "launch",
            "program": "${workspaceFolder}/backend/main.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/backend",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/backend"
            }
        }
    ]
}
```

## 📞 Sorun Devam Ederse

Kurulum scriptleri çalıştırdıktan sonra hala sorun varsa:

1. PowerShell'i yönetici olarak çalıştırın
2. `Get-ExecutionPolicy` komutuyla policy kontrol edin
3. Gerekirse: `Set-ExecutionPolicy RemoteSigned`
4. Python ve Node.js PATH'lerini manuel kontrol edin
5. VS Code'u yeniden başlatın
