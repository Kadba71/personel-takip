## Hızlı Başlangıç

- Backend: `cd backend`; sanal ortamı etkinleştirin ve `python main.py` çalıştırın. Varsayılan host `127.0.0.1`, port `8000` (config.json veya .env ile `8002` yapabilirsiniz).
- Frontend (vanilla): Backend tarafından `http://127.0.0.1:<port>/` adresinden servis edilir.
- React (opsiyonel): `cd frontend-react; npm install; npm run dev`.

Notlar:
- `.env.example`'ı `.env` olarak kopyalayın ve `PERSONEL_TAKIP_SERVER__PORT=8002` ayarlayın. Frontend aynı origin'i kullanır.
- İlk kullanıcı yoksa debug modunda `admin/admin` otomatik oluşur. Üretimde env ile `ADMIN_USERNAME` ve `ADMIN_PASSWORD` verin.
- Health endpoint artık public: `/api/health`.
