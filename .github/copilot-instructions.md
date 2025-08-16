# Copilot Instructions for Personel Takip Project

## Project Architecture
- **Monorepo structure**: Contains `backend` (FastAPI), `frontend` (vanilla JS/HTML/CSS), and `frontend-react` (React + TypeScript + Vite) directories.
- **Backend**: `backend/main.py` is the FastAPI app. It exposes REST endpoints for personnel management, daily records, analytics, and Excel import/export. Data is currently in-memory; SQLAlchemy integration is planned.
- **Frontend**: `frontend/` is a static site. `frontend-react/` is a Vite-based React app with TypeScript and ESLint config.
- **Shared**: `shared/` is present for future code/data sharing, but currently unused.

## Backend Patterns
- **Endpoints**: All API routes are defined in `main.py`. Example: `/api/personnel`, `/api/daily-records`, `/api/analytics/summary`.
- **Data**: Uses Python lists for `personnel_data` and `daily_records_data`. Each personnel and record is a dict with nested fields.
- **Excel Export**: Uses pandas and xlsxwriter to export personnel and daily records to Excel. See `export_to_excel` endpoint.
- **Error Handling**: Uses FastAPI's `HTTPException` for error responses. Print statements are used for debugging in CRUD endpoints.
- **Performance Calculation**: Daily record endpoints calculate `performance_score` based on targets and values, with custom logic for positive/negative metrics.
- **Security**: HTTPBearer is set up but not enforced on endpoints yet.

## Frontend-React Patterns
- **Vite + React + TypeScript**: See `frontend-react/README.md` for ESLint and Vite config details.
- **ESLint**: Uses type-aware linting. See README for recommended config and plugin usage.
- **Entry point**: `src/main.tsx` and `src/App.tsx`.

## Developer Workflows
- **Run Backend**: From `backend/`, use:
  ```powershell
  uvicorn main:app --reload
  ```
  Or run as a script (see `if __name__ == "__main__"` block).
- **Run Frontend-React**: From `frontend-react/`, use:
  ```powershell
  npm install
  npm run dev
  ```
- **Excel Export**: Call `/api/export/excel` endpoint; file is saved in `backend/`.

## Conventions & Integration
- **Endpoints**: All backend endpoints return a dict with `success`, `data`, and `timestamp` fields.
- **Date Handling**: Dates are strings in `YYYY-MM-DD` format.
- **Frontend Integration**: CORS is enabled for localhost ports 3000, 5173, 8080 for frontend dev.
- **No database yet**: All data is in-memory; changes are not persisted.

## External Dependencies
- **Backend**: FastAPI, pandas, xlsxwriter, uvicorn
- **Frontend-React**: React, Vite, TypeScript, ESLint

## Key Files
- `backend/main.py`: All backend logic and endpoints
- `frontend-react/README.md`: React/Vite/ESLint config and conventions
- `frontend/app.js`, `frontend/index.html`: Vanilla JS frontend

---

If you add new endpoints, keep response format consistent. For analytics, follow the summary/trend pattern in `main.py`. For React, follow Vite/TypeScript conventions in `frontend-react/README.md`.

Please review and suggest edits if any section is unclear or missing important project-specific details.
