// Prefer same-origin API; allow override via window.API_BASE_URL. Fallback to localhost:8002.
const API_BASE_URL = (typeof window !== 'undefined' && window.API_BASE_URL)
    || (typeof window !== 'undefined' ? `${window.location.origin}` : '')
    || 'http://127.0.0.1:8002';
// Auth token helpers + global fetch wrapper
function getAuthToken(){ try { return localStorage.getItem('apiAuthToken') || ''; } catch { return ''; } }
function setAuthToken(t){ try { if (t) localStorage.setItem('apiAuthToken', t); else localStorage.removeItem('apiAuthToken'); } catch {} }
function isLoggedIn(){ return !!getAuthToken(); }
// Attach Authorization header automatically for API requests
(() => {
    try {
        const originalFetch = window.fetch ? window.fetch.bind(window) : null;
        if (!originalFetch) return;
        const shouldAttach = (url) => {
            try {
                if (typeof url !== 'string') return false;
                if (url.startsWith('/api/')) return true;
                if (url.startsWith(API_BASE_URL)) return true;
                return false;
            } catch { return false; }
        };
        const ignore401 = (url) => {
            try {
                if (typeof url !== 'string') return true;
                // Don't auto-redirect on login/me endpoints
                return url.includes('/api/auth/login') || url.includes('/api/auth/me');
            } catch { return true; }
        };
        window.fetch = function(input, init){
            let urlStr = '';
            if (typeof input === 'string') urlStr = input;
            else if (input && typeof input.url === 'string') urlStr = input.url;
            let finalInit = init || {};
            try {
                if (shouldAttach(urlStr)) {
                    const token = getAuthToken();
                    if (token) {
                        const headers = new Headers(finalInit.headers || {});
                        if (!headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
                        finalInit = { ...finalInit, headers };
                    }
                }
            } catch {}
            const p = originalFetch(input, finalInit);
            return p.then((resp) => {
                try {
                    if (resp && resp.status === 401 && shouldAttach(urlStr) && !ignore401(urlStr)) {
                        // Token invalid/expired: clear and route to login
                        setAuthToken('');
                        try { updateAuthNavUI && updateAuthNavUI(); } catch {}
                        try { typeof showNotification === 'function' && showNotification('Oturum süresi doldu. Lütfen tekrar giriş yapın.', 'warning', 'Uyarı'); } catch {}
                        try { typeof showPage === 'function' && showPage('login'); } catch {}
                    }
                } catch {}
                return resp;
            });
        };
        console.log('🔐 Auth fetch wrapper enabled');
    } catch (e) { console.warn('Auth wrapper init failed', e); }
})();

// === Globals and helpers (prevent ReferenceError) ===
let toastManager = null;
let globalEventListenerAttached = false;
let globalClickHandlerAttached = false;
// Dashboard summary view state (filters, sort, rows)
const dashboardSummaryState = {
    startDate: '',
    endDate: '',
    rows: [],
    sortKey: 'default', // default = team then name
    sortDir: 'asc'
};
// General dashboard computed state cache
const dashboardState = {
    lastComputedAt: 0,
    lastTargetsAddedCount: 0
};

// === Toast Notification System ===
class ToastManager {
    constructor() {
        this.container = document.getElementById('toastContainer');
        this.toasts = [];
    }
    show(message, type = 'success', title = '', duration = 4000) {
        if (!this.container) {
            // Try to late-bind container if DOM not ready at construct time
            this.container = document.getElementById('toastContainer');
            if (!this.container) return;
        }
        const toast = this.createToast(message, type, title, duration);
        this.container.appendChild(toast);
        this.toasts.push(toast);
        requestAnimationFrame(() => toast.classList.add('show'));
        setTimeout(() => this.hide(toast), duration);
        return toast;
    }
    createToast(message, type, title) {
        const escapeHTML = (str) => String(str || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        const toast = document.createElement('div');
        const safeType = typeof type === 'string' ? type : 'info';
        toast.className = `toast ${safeType}`;
        const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
        const icon = icons[safeType] || icons.info;
        const safeTitle = title ? `<strong>${escapeHTML(title)}</strong><br>` : '';
        const safeMessage = escapeHTML(message);
        toast.innerHTML = `
            <span class="toast-icon">${icon}</span>
            <div class="toast-message">${safeTitle}${safeMessage}</div>
            <button class="toast-close" aria-label="Kapat">&times;</button>
        `;
        const closeBtn = toast.querySelector('.toast-close');
        if (closeBtn) closeBtn.addEventListener('click', () => this.hide(toast));
        return toast;
    }
    hide(toast) {
        if (!toast) return;
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
            this.toasts = this.toasts.filter(t => t !== toast);
        }, 300);
    }
}

// Button click helper with simple locking and visual feedback
const buttonStates = new Map();
function handleButtonClick(buttonElement, callback, minDelay = 600) {
    if (!buttonElement || typeof callback !== 'function') return false;
    const id = buttonElement.id || `btn-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    if (buttonStates.get(id)) return false;
    buttonStates.set(id, true);
    const originalText = buttonElement.textContent;
    const originalDisabled = buttonElement.disabled;
    buttonElement.disabled = true;
    buttonElement.textContent = `${originalText} ⏳`;
    buttonElement.classList.add('btn-loading');
    const finalize = () => {
        buttonElement.textContent = originalText;
        buttonElement.disabled = originalDisabled;
        buttonElement.classList.remove('btn-loading');
        buttonStates.delete(id);
    };
    (async () => {
        try { await callback(); }
        catch (e) { try { showNotification('İşlem sırasında hata oluştu', 'error', 'Hata'); } catch {} }
        finally { setTimeout(finalize, minDelay); }
    })();
    return true;
}

// FORM VALIDATION SCHEMAS
const validationSchemas = {
    personnel: {
        name: { rules: ['required'], displayName: 'Personel Adı' },
        username: { rules: ['required'], displayName: 'Kullanıcı Adı' },
        email: { rules: ['email'], displayName: 'E-posta' },
        hire_date: { rules: ['required', 'date'], displayName: 'İşe Giriş Tarihi' },
        team: { rules: ['required'], displayName: 'Ekip' }
    },
    
    performance: {
        performance_personnel_id: { rules: ['required'], displayName: 'Personel' },
        performance_date: { rules: ['required', 'date'], displayName: 'Tarih' },
        member_count: { rules: ['positiveNumber'], displayName: 'Üye Adedi' },
        whatsapp_count: { rules: ['positiveNumber'], displayName: 'WhatsApp Adedi' },
        device_count: { rules: ['positiveNumber'], displayName: 'Cihaz Adedi' },
        unanswered_count: { rules: ['positiveNumber'], displayName: 'Cevapsız Adedi' }
    },
    
    trainingFeedback: {
        training_feedback_personnel_id: { rules: ['required'], displayName: 'Personel' },
        training_feedback_date: { rules: ['required', 'date'], displayName: 'Tarih' }
    },
    
    dailyRecord: {
        personnel_id: { rules: ['required'], displayName: 'Personel' },
        record_date: { rules: ['required', 'date'], displayName: 'Tarih' },
        call_number: { rules: ['required'], displayName: 'Çağrı Numarası' },
        score: { rules: ['required', 'positiveNumber'], displayName: 'Puan' },
        notes: { rules: ['required'], displayName: 'Geribildirim' }
    },
    targets: {
        target_personnel_id: { rules: ['required'], displayName: 'Personel' },
        target_start_date: { rules: ['required', 'date'], displayName: 'Başlangıç Tarihi' },
        target_end_date: { rules: ['required', 'date'], displayName: 'Bitiş Tarihi' },
        target_member_count: { rules: ['required', 'positiveNumber'], displayName: 'Hedeflenen Üye Sayısı' }
    },
    afterHours: {
        after_hours_personnel_id: { rules: ['required'], displayName: 'Personel' },
        after_hours_date: { rules: ['required', 'date'], displayName: 'Tarih' },
        after_hours_call_count: { rules: ['positiveNumber'], displayName: 'Arama Adedi' },
        after_hours_talk_duration: { rules: ['positiveNumber'], displayName: 'Konuşma Süresi (dk)' },
        after_hours_member_count: { rules: ['positiveNumber'], displayName: 'Üye Adedi' }
    },
    attendanceOverride: {
        attendance_override_personnel_id: { rules: ['required'], displayName: 'Personel' },
        attendance_override_date: { rules: ['required', 'date'], displayName: 'Tarih' },
        attendance_override_value: { rules: ['required'], displayName: 'Değer' }
    },
    warningCut: {
        warning_cut_personnel_id: { rules: ['required'], displayName: 'Personel' },
        warning_cut_date: { rules: ['required', 'date'], displayName: 'Tarih' },
        warning_cut_type: { rules: ['required'], displayName: 'Uyarı/Kesinti' },
        warning_cut_count: { rules: ['required', 'positiveNumber'], displayName: 'Adet' }
    }
};

// Minimal form validator used by handleFormSubmitInline
// Supports rules used in validationSchemas: required, date, email, positiveNumber
const FormValidator = {
    validateForm(data, schema) {
        try {
            const errors = [];
            if (!schema || typeof schema !== 'object') return errors;
            const isEmpty = (v) => v == null || (typeof v === 'string' && v.trim() === '');
            const isValidDate = (v) => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(v);
            const isEmail = (v) => typeof v === 'string' && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
            const isPosNum = (v) => {
                if (isEmpty(v)) return true; // allow empty unless also required
                const n = Number(v);
                return Number.isFinite(n) && n >= 0;
            };
            for (const [field, cfg] of Object.entries(schema)) {
                const display = (cfg && cfg.displayName) || field;
                const rules = (cfg && cfg.rules) || [];
                const raw = data ? (data[field]) : undefined;
                for (const rule of rules) {
                    if (rule === 'required') {
                        if (isEmpty(raw)) errors.push(`${display} zorunludur`);
                    } else if (rule === 'date') {
                        if (!isEmpty(raw) && !isValidDate(String(raw))) errors.push(`${display} geçerli bir tarih olmalıdır (YYYY-MM-DD)`);
                    } else if (rule === 'email') {
                        if (!isEmpty(raw) && !isEmail(String(raw))) errors.push(`${display} geçerli bir e-posta olmalıdır`);
                    } else if (rule === 'positiveNumber') {
                        if (!isPosNum(raw)) errors.push(`${display} 0 veya daha büyük bir sayı olmalıdır`);
                    }
                }
            }
            return errors;
        } catch (e) {
            // Fail-open: if validator has an issue, do not block submission
            console.warn('FormValidator error, skipping validation:', e);
            return [];
        }
    }
};

// INLINE FORM SUBMIT HANDLER - CACHE bypass için
async function handleFormSubmitInline(form) {
    const originalElement = form;
    // Normalize: ensure we have the actual FORM element (sometimes a child/input is passed)
    let resolvedForm = form;
    if (resolvedForm && resolvedForm.tagName !== 'FORM') {
        resolvedForm = resolvedForm.form || resolvedForm.closest && resolvedForm.closest('form');
        // If still not found, try attribute 'form' on the original target
        if (!resolvedForm && form && typeof form.getAttribute === 'function') {
            const relatedFormId = form.getAttribute('form');
            if (relatedFormId) {
                resolvedForm = document.getElementById(relatedFormId);
            }
        }
    }
    // Fallback: if still not a FORM, attempt to infer by common edit context (e.g., editPersonnelForm)
    if (!resolvedForm || resolvedForm.tagName !== 'FORM') {
        const maybeEditPersonnelForm = document.getElementById('editPersonnelForm');
        if (maybeEditPersonnelForm) {
            resolvedForm = maybeEditPersonnelForm;
        }
    }
    if (!resolvedForm || resolvedForm.tagName !== 'FORM') {
        console.warn('⚠️ Could not resolve a FORM element for inline submit. Aborting. Got:', form);
        return;
    }

    // Safely resolve the FORM id; forms can have a control named "id" which shadows form.id
    let resolvedFormId = (typeof resolvedForm.getAttribute === 'function' ? resolvedForm.getAttribute('id') : resolvedForm.id) || '';
    // If form id is empty but we know its context (contains #editPersonnelId), assign a synthetic id to aid logic
    if (!resolvedFormId && resolvedForm.querySelector && resolvedForm.querySelector('#editPersonnelId')) {
        resolvedForm.setAttribute('id', 'editPersonnelForm');
        resolvedFormId = 'editPersonnelForm';
    }
    console.log('📝 INLINE Handling form submit:', resolvedFormId);
    console.log('🧭 RESOLVED FORM:', { tag: resolvedForm.tagName, id: resolvedFormId });
    
    // Special-case: Login form is handled here directly to avoid generic routing
    if (resolvedFormId === 'loginForm') {
        try {
            const submitBtn = document.getElementById('loginSubmitBtn');
            const u = (document.getElementById('login_username')?.value || '').toString().trim();
            const p = (document.getElementById('login_password')?.value || '').toString().trim();
            if (!u || !p) { showNotification('Kullanıcı adı ve şifre gerekli', 'warning', 'Uyarı'); return; }
            if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Giriş Yap ⏳'; }
            const resp = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: u, password: p })
            });
            if (!resp.ok) {
                const err = await resp.json().catch(()=>({}));
                showNotification(err.detail || 'Giriş başarısız', 'error', 'Hata');
                return;
            }
            const json = await resp.json();
            const token = json?.data?.token || '';
            if (!token) { showNotification('Token alınamadı', 'error', 'Hata'); return; }
            setAuthToken(token);
            showNotification('Giriş başarılı', 'success', 'Başarılı');
            showPage('dashboard');
        } catch (e) {
            showNotification('Sunucuya ulaşılamadı', 'error', 'Hata');
        } finally {
            const submitBtn = document.getElementById('loginSubmitBtn');
            if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Giriş Yap'; }
        }
        return;
    }

    const formData = new FormData(resolvedForm);
    const data = Object.fromEntries(formData.entries());
    // Normalized copy for validation/payload overrides
    let dataForValidation = { ...data };
    let payloadData = { ...data };
    
    console.log('📋 Form data:', data);
    
    // GET VALIDATION SCHEMA
    let validationSchema;
    let apiEndpoint;
    let successMessage;
    let errorMessage;
    
    if (resolvedFormId === 'addPersonnelForm') {
        validationSchema = validationSchemas.personnel;
        apiEndpoint = '/api/personnel';
        successMessage = `${data.name} başarıyla eklendi`;
        errorMessage = 'Personel Ekleme Hatası';

    } else if (resolvedFormId === 'addPerformanceForm') {
        validationSchema = validationSchemas.performance;
        apiEndpoint = '/api/performance-records';
        successMessage = 'Performans kaydı başarıyla eklendi';
        errorMessage = 'Performans Ekleme Hatası';
    } else if (resolvedFormId === 'editPerformanceForm') {
        validationSchema = validationSchemas.performance;
        const recordId = document.getElementById('editPerformanceId').value;
        apiEndpoint = `/api/performance-records/${recordId}`;
        successMessage = 'Performans kaydı başarıyla güncellendi';
        errorMessage = 'Performans Güncelleme Hatası';
    } else if (resolvedFormId === 'addTrainingFeedbackForm') {
        validationSchema = validationSchemas.trainingFeedback;
        apiEndpoint = '/api/training-feedback';
        successMessage = 'Eğitim geri bildirimi başarıyla eklendi';
        errorMessage = 'Eğitim Geri Bildirimi Ekleme Hatası';
    } else if (resolvedFormId === 'editPersonnelForm') {
        // Personel düzenleme formu
        validationSchema = validationSchemas.personnel;
        const recordId = (data.id 
            || document.getElementById('editPersonnelId')?.value 
            || resolvedForm.querySelector('#editPersonnelId')?.value 
            || '').toString();
        if (!recordId) {
            console.warn('⚠️ editPersonnelForm için ID bulunamadı');
        }
        apiEndpoint = `/api/personnel/${recordId}`;
        successMessage = 'Personel bilgileri başarıyla güncellendi';
        errorMessage = 'Personel Güncelleme Hatası';
    } else if (resolvedFormId === 'editTrainingFeedbackForm') {
        validationSchema = validationSchemas.trainingFeedback;
        const recordId = document.getElementById('editTrainingFeedbackId').value;
        apiEndpoint = `/api/training-feedback/${recordId}`;
        successMessage = 'Eğitim geri bildirimi başarıyla güncellendi';
        errorMessage = 'Eğitim Geri Bildirimi Güncelleme Hatası';
    } else if (resolvedFormId === 'editDailyRecordForm') {
        validationSchema = validationSchemas.dailyRecord;
        const recordId = (document.getElementById('editDailyRecordId')?.value || data.id || '').toString();
        apiEndpoint = `/api/daily-records/${recordId}`;
        successMessage = 'Günlük kayıt başarıyla güncellendi';
        errorMessage = 'Günlük Kayıt Güncelleme Hatası';
    } else if (resolvedFormId === 'addRecordForm') {
        validationSchema = validationSchemas.dailyRecord;
        apiEndpoint = '/api/daily-records';
        successMessage = 'Günlük kayıt başarıyla eklendi';
        errorMessage = 'Günlük Kayıt Ekleme Hatası';
    } else if (resolvedFormId === 'setTargetsForm') {
        // Hedef belirleme formu
        validationSchema = validationSchemas.targets;
        apiEndpoint = '/api/targets';
        successMessage = 'Hedef başarıyla eklendi';
        errorMessage = 'Hedef Ekleme Hatası';
    } else if (!resolvedFormId && resolvedForm.querySelector && resolvedForm.querySelector('#editPersonnelId')) {
        // Heuristic fallback: treat as editPersonnelForm if it contains the hidden id field
        console.log('🧠 Heuristic match: treating as editPersonnelForm based on #editPersonnelId presence');
        validationSchema = validationSchemas.personnel;
        const recordId = (data.id 
            || resolvedForm.querySelector('#editPersonnelId')?.value 
            || '').toString();
        apiEndpoint = `/api/personnel/${recordId}`;
        successMessage = 'Personel bilgileri başarıyla güncellendi';
        errorMessage = 'Personel Güncelleme Hatası';
    } else if (resolvedFormId === 'afterHoursForm') {
        // Mesai sonrası kayıt formu (add or edit based on hidden id)
        validationSchema = validationSchemas.afterHours;
        const editId = (data.afterHoursEditId || resolvedForm.querySelector?.('#afterHoursEditId')?.value || '').toString();
        if (editId) {
            apiEndpoint = `/api/after-hours/${editId}`;
            successMessage = 'Mesai sonrası kaydı güncellendi';
            errorMessage = 'Mesai Sonrası Güncelleme Hatası';
        } else {
            apiEndpoint = '/api/after-hours';
            successMessage = 'Mesai sonrası kaydı eklendi';
            errorMessage = 'Mesai Sonrası Ekleme Hatası';
        }
        // Map fields for validation (already matches) and payload to backend keys
        dataForValidation = { ...data };
        payloadData = {
            date: (data.after_hours_date || '').toString(),
            personnel_id: parseInt(data.after_hours_personnel_id || 0) || 0,
            call_count: parseInt(data.after_hours_call_count || 0) || 0,
            talk_duration: parseInt(data.after_hours_talk_duration || 0) || 0,
            member_count: parseInt(data.after_hours_member_count || 0) || 0,
            notes: (data.after_hours_notes || '').toString()
        };
    } else if (resolvedFormId === 'attendanceOverrideForm') {
        // Puantaj/İzin Overrides (add or edit)
        validationSchema = validationSchemas.attendanceOverride;
        const editId = (data.attendance_override_edit_id || '').toString();
        if (editId) {
            apiEndpoint = `/api/attendance/${editId}`;
            successMessage = 'Puantaj kaydı güncellendi';
            errorMessage = 'Puantaj Güncelleme Hatası';
        } else {
            apiEndpoint = '/api/attendance';
            successMessage = 'Puantaj kaydı eklendi';
            errorMessage = 'Puantaj Ekleme Hatası';
        }
        dataForValidation = { ...data };
        // Support 'Haftalık izin (1)' option which we encode as value=1 and leave_type='weekly'
        const rawVal = (data.attendance_override_value || '').toString();
        const isWeekly = rawVal === '1-week';
        payloadData = {
            date: (data.attendance_override_date || '').toString(),
            personnel_id: parseInt(data.attendance_override_personnel_id || 0) || 0,
            value: isWeekly ? 1 : parseFloat(rawVal),
            notes: (data.attendance_override_notes || '').toString(),
            ...(isWeekly ? { leave_type: 'weekly', period: 'weekly' } : {})
        };
        // If no explicit edit id but an override exists for this (personnel,date), switch to PUT
        try {
            if (!editId) {
                const pid = payloadData.personnel_id;
                const dt = payloadData.date;
                const ovMap = window.__attendanceOverridesMap || new Map();
                const key = `${pid}|${dt}`;
                const existing = ovMap.get ? ovMap.get(key) : null;
                if (existing && existing.id != null) {
                    apiEndpoint = `/api/attendance/${existing.id}`;
                    // Force PUT method later
                    resolvedForm.setAttribute('data-force-put', 'true');
                } else {
                    resolvedForm.removeAttribute('data-force-put');
                }
            }
        } catch {}
    } else if (resolvedFormId === 'warningCutForm') {
        // UYARI - KESİNTİ kayıtları (yeni endpoint)
        validationSchema = validationSchemas.warningCut;
        // If hidden edit id is provided, update existing record
        const editId = (data.warning_cut_edit_id || resolvedForm.querySelector?.('#warning_cut_edit_id')?.value || '').toString();
        if (editId) {
            apiEndpoint = `/api/warnings-cuts/${editId}`;
            successMessage = 'Uyarı/Kesinti kaydı güncellendi';
            errorMessage = 'Uyarı/Kesinti Güncelleme Hatası';
        } else {
            apiEndpoint = '/api/warnings-cuts';
            successMessage = 'Uyarı/Kesinti kaydı eklendi';
            errorMessage = 'Uyarı/Kesinti Ekleme Hatası';
        }
        dataForValidation = { ...data };
        payloadData = {
            date: (data.warning_cut_date || '').toString(),
            personnel_id: parseInt(data.warning_cut_personnel_id || 0) || 0,
            warning_interruption_type: (data.warning_cut_type || '').toString(),
            warning_interruption_subject: (data.warning_cut_subject || '').toString(),
            warning_interruption_count: parseInt(data.warning_cut_count || 1) || 1
        };
    } else {
        // Data-shape based fallback routing
        const keys = Object.keys(data || {});
        const isPersonnelShape = ['name','username','hire_date','team'].every(k => keys.includes(k));
        if (isPersonnelShape) {
            console.log('🧩 Fallback by data shape: treating as personnel form');
            validationSchema = validationSchemas.personnel;
            const recordId = (data.id 
                || resolvedForm.querySelector?.('#editPersonnelId')?.value 
                || (originalElement && originalElement.id === 'editPersonnelId' ? originalElement.value : '')
                || '').toString();
            apiEndpoint = recordId ? `/api/personnel/${recordId}` : '/api/personnel';
            successMessage = recordId ? 'Personel bilgileri başarıyla güncellendi' : `${data.name} başarıyla eklendi`;
            errorMessage = recordId ? 'Personel Güncelleme Hatası' : 'Personel Ekleme Hatası';
        } else {
            console.log('⚠️ Unknown form type:', resolvedFormId);
            console.log('🌐 Fallback debug outerHTML (trimmed):', resolvedForm && resolvedForm.outerHTML ? resolvedForm.outerHTML.slice(0, 200) + '…' : 'n/a');
            return;
        }
    }
    
    // Map edit form fields to validation schema keys when needed
    if (resolvedFormId === 'editPerformanceForm') {
        // Validation expects performance_personnel_id & performance_date
        dataForValidation = {
            ...dataForValidation,
            performance_personnel_id: data.editPerformancePersonnel || document.getElementById('editPerformancePersonnel')?.value,
            performance_date: data.editPerformanceDate || document.getElementById('editPerformanceDate')?.value
        };
        // Backend expects canonical keys for update
        payloadData = {
            personnel_id: parseInt(data.editPerformancePersonnel || document.getElementById('editPerformancePersonnel')?.value || 0) || 0,
            // include date so backend can update it
            date: (data.editPerformanceDate || document.getElementById('editPerformanceDate')?.value || '').toString(),
            member_count: parseInt(data.editPerformanceMemberCount || 0) || 0,
            whatsapp_count: parseInt(data.editPerformanceWhatsappCount || 0) || 0,
            device_count: parseInt(data.editPerformanceDeviceCount || 0) || 0,
            unanswered_count: parseInt(data.editPerformanceUnansweredCount || 0) || 0,
            knowledge_duel_result: (data.editPerformanceKnowledgeDuelResult ?? '').toString(),
            reward_penalty: (data.editPerformanceRewardPenalty ?? '').toString()
        };
    } else if (resolvedFormId === 'editTrainingFeedbackForm') {
        // Validation schema already matches name attributes; just ensure values exist
        dataForValidation = {
            ...dataForValidation,
            training_feedback_personnel_id: data.training_feedback_personnel_id || document.getElementById('edit_training_feedback_personnel_id')?.value,
            training_feedback_date: data.training_feedback_date || document.getElementById('edit_training_feedback_date')?.value
        };
        // Backend update endpoint expects different keys than add; map accordingly
        payloadData = {
            personnel_id: parseInt(data.training_feedback_personnel_id || document.getElementById('edit_training_feedback_personnel_id')?.value || 0) || 0,
            date: data.training_feedback_date || document.getElementById('edit_training_feedback_date')?.value || '',
            feedback_count: parseInt(data.feedback_count || 0) || 0,
            feedback_subject: data.feedback_subject || '',
            general_training_count: parseInt(data.general_training_count || 0) || 0,
            general_training_subject: data.general_training_subject || '',
            // Backend will map one_on_one_* to personal_training_*
            one_on_one_training_count: parseInt(data.one_on_one_training_count || 0) || 0,
            one_on_one_training_subject: data.one_on_one_training_subject || '',
            notes: data.notes || ''
        };
    } else if (resolvedFormId === 'editDailyRecordForm') {
        // Daily record validation matches schema keys; ensure they exist
        dataForValidation = {
            personnel_id: data.personnel_id || document.querySelector('#editDailyRecordPersonnel')?.value,
            record_date: data.record_date || document.querySelector('#editDailyRecordDate')?.value,
            call_number: data.call_number || document.querySelector('#editDailyRecordCallNumber')?.value,
            score: data.score || document.querySelector('#editDailyRecordScore')?.value,
            notes: data.notes || document.querySelector('#editDailyRecordNotes')?.value
        };
        payloadData = {
            date: dataForValidation.record_date,
            personnel_id: parseInt(dataForValidation.personnel_id || 0) || 0,
            call_number: (dataForValidation.call_number || '').toString(),
            score: parseInt(dataForValidation.score || 0) || 0,
            notes: (dataForValidation.notes || '').toString()
        };
    }

    // VALIDATE FORM
    console.log('🔍 Validating form data:', data);
    console.log('🔍 Using validation schema:', validationSchema);
    const validationErrors = FormValidator.validateForm(dataForValidation, validationSchema);
    
    if (validationErrors.length > 0) {
        console.log('⚠️ Validation errors:', validationErrors);
        console.log('⚠️ Individual errors:');
        validationErrors.forEach((error, index) => {
            console.log(`  ${index + 1}. ${error}`);
        });
        showNotification(
            validationErrors.join('\n'),
            'error',
            'Form Doğrulama Hatası'
        );
        return;
    }
    
    console.log('✅ Form validation passed:', data);
    
    // API CALL
    try {
    const hasId = !!(data?.id || resolvedForm.querySelector?.('#editPersonnelId')?.value);
    const formIdStr = typeof resolvedFormId === 'string' ? resolvedFormId : String(resolvedFormId || '');
    let method = (formIdStr.startsWith('edit') || hasId) ? 'PUT' : 'POST';
        console.log(`📤 Sending ${method} to ${apiEndpoint}`, { formIdStr, hasId });
        // Choose payload per form
    const bodyToSend = (resolvedFormId === 'editPerformanceForm' || resolvedFormId === 'editTrainingFeedbackForm' || resolvedFormId === 'editDailyRecordForm' || resolvedFormId === 'afterHoursForm' || resolvedFormId === 'attendanceOverrideForm' || resolvedFormId === 'warningCutForm')
            ? payloadData
            : (resolvedFormId === 'setTargetsForm'
                ? {
                    personnel_id: parseInt(data.target_personnel_id || 0) || 0,
                    // Backend expects target_value and optional target_type (default 'uye_adedi')
                    target_value: parseInt(data.target_member_count || 0) || 0,
                    target_type: 'uye_adedi',
                    start_date: data.target_start_date || '',
                    end_date: data.target_end_date || ''
                }
                : data);
    // Set PUT route for Targets when editing
    if (resolvedFormId === 'setTargetsForm') {
        const editId = (data.editTargetId || resolvedForm.querySelector?.('#editTargetId')?.value || '').toString();
        if (editId) {
            method = 'PUT';
            apiEndpoint = `/api/targets/${editId}`;
            successMessage = 'Hedef başarıyla güncellendi';
            errorMessage = 'Hedef Güncelleme Hatası';
        }
    }
    // If after-hours edit id present, force method to PUT
    if (resolvedFormId === 'afterHoursForm') {
            const ahEditId = (resolvedForm.querySelector?.('#afterHoursEditId')?.value || data.afterHoursEditId || '').toString();
            if (ahEditId) {
                method = 'PUT';
            }
        }
    // If attendance override edit id present, force method to PUT
    if (resolvedFormId === 'attendanceOverrideForm') {
            const aoEditId = (resolvedForm.querySelector?.('#attendance_override_edit_id')?.value || data.attendance_override_edit_id || '').toString();
            if (aoEditId || resolvedForm.getAttribute('data-force-put') === 'true') {
                method = 'PUT';
            }
        }
    // If warning/cut edit id present, force method to PUT
    if (resolvedFormId === 'warningCutForm') {
            const wcEditId = (resolvedForm.querySelector?.('#warning_cut_edit_id')?.value || data.warning_cut_edit_id || '').toString();
            if (wcEditId) {
                method = 'PUT';
            }
        }
    // Build headers; auth is auto-attached by global fetch wrapper
    const headers = { 'Content-Type': 'application/json' };
    const response = await fetch(`${API_BASE_URL}${apiEndpoint}`, {
            method: method,
            headers,
            body: JSON.stringify(bodyToSend)
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Success:', result);
            
            // Show success notification
            showNotification(successMessage, 'success', 'İşlem Başarılı');
            
            // FORCE CLOSE MODALS - Use multiple approaches
            console.log('🚪 Force closing modals...');
            try {
                if (typeof closeModals === 'function') {
                    closeModals();
                    console.log('✅ Modals closed via closeModals()');
                } else if (typeof window.closeModals === 'function') {
                    window.closeModals();
                    console.log('✅ Modals closed via window.closeModals()');
                } else {
                    // Fallback - manual close
                    console.log('⚠️ closeModals function not found, using fallback');
                    document.querySelectorAll('.modal-overlay').forEach(overlay => {
                        overlay.style.display = 'none';
                        overlay.classList.remove('active');
                    });
                    document.querySelectorAll('.modal').forEach(modal => {
                        modal.style.display = 'none';
                        modal.classList.remove('active');
                    });
                    document.body.style.overflow = 'auto';
                    document.body.classList.remove('modal-open');
                    console.log('✅ Modals closed via fallback method');
                }
            } catch (error) {
                console.error('❌ Error closing modals:', error);
                // Fallback - manual close
                document.querySelectorAll('.modal-overlay').forEach(overlay => {
                    overlay.style.display = 'none';
                    overlay.classList.remove('active');
                });
                console.log('✅ Emergency modal close executed');
            }
            
            // Reset the resolved form (not the original element which may be an input/button)
            resolvedForm.reset();
            // Clear editTargetId if present so next open is add-mode
            try { const hid = document.getElementById('editTargetId'); if (hid) hid.value = ''; } catch {}
            console.log('✅ Form reset completed');
            // Extra debug to ensure branch matching below
            console.log('🧭 Post-reset resolvedForm.id:', resolvedForm && resolvedForm.id);

            // Refresh appropriate content based on form type
            if (resolvedFormId === 'addPersonnelForm') {
                console.log('📋 Personnel form submitted successfully, refreshing content...');
                showPage('personnel');
                console.log('🔄 Waiting for page transition, then calling updatePersonnelContent()...');
                // Wait a bit for page transition to complete
                setTimeout(() => {
                    console.log('🔄 Now calling updatePersonnelContent()...');
                    updatePersonnelContent();
                    console.log('✅ Personnel content update completed');
                    // Also refresh dashboard quick stats
                    try { updateDashboardContent(); } catch {}
                }, 100);
            } else if (resolvedFormId === 'editPersonnelForm') {
                console.log('✏️ Personnel edit form submitted successfully, applying optimistic row update...');
                try {
                    const editedId = data.id || document.getElementById('editPersonnelId')?.value;
                    if (editedId) {
                        const row = document.querySelector(`tr[data-personnel-id="${editedId}"]`);
                        if (row) {
                            const cells = row.querySelectorAll('td');
                            // Expected order: name, username, email, team badge cell, hire_date, reference, promotion_date, actions
                            if (cells.length >= 8) {
                                cells[0].textContent = data.name || '-';
                                cells[1].textContent = data.username || '-';
                                cells[2].textContent = data.email || '-';
                                // Team badge
                                const teamSpan = document.createElement('span');
                                teamSpan.className = `team-badge ${data.team === 'As Ekip' ? 'team-as' : 'team-paf'}`;
                                teamSpan.textContent = data.team || '-';
                                cells[3].innerHTML = '';
                                cells[3].appendChild(teamSpan);
                                cells[4].textContent = data.hire_date || '-';
                                cells[5].textContent = data.reference || '-';
                                cells[6].textContent = data.promotion_date || '-';
                                row.classList.add('highlight-updated');
                                setTimeout(() => row.classList.remove('highlight-updated'), 2000);
                                console.log('✅ Optimistic row update applied for personnel ID', editedId);
                                // Immediate table refresh (plus background delayed refresh below)
                                updatePersonnelContent();
                                try { updateDashboardContent(); } catch {}
                            }
                        } else {
                            console.log('ℹ️ Row not found for optimistic update, triggering full refresh');
                            updatePersonnelContent();
                            try { updateDashboardContent(); } catch {}
                        }
                    }
                    // Schedule a background refetch to reconcile ordering or new server data
                    setTimeout(() => {
                        updatePersonnelContent().catch(err => console.warn('Background personnel refresh failed:', err));
                        try { updateDashboardContent(); } catch {}
                    }, 800);
                } catch (optErr) {
                    console.error('❌ Optimistic update failed, fallback to full refresh:', optErr);
                    updatePersonnelContent();
                    try { updateDashboardContent(); } catch {}
                }
            } else if (typeof apiEndpoint === 'string' && apiEndpoint.startsWith('/api/personnel')) {
                // Defensive fallback: if this was a personnel API call but branch didn't match,
                // still refresh the personnel list immediately to reflect changes.
                console.log('🛡️ Fallback: refreshing personnel due to personnel API call');
                updatePersonnelContent();
                try { updateDashboardContent(); } catch {}
                setTimeout(() => {
                    updatePersonnelContent().catch(err => console.warn('Background personnel refresh failed:', err));
                    try { updateDashboardContent(); } catch {}
                }, 800);
            } else if (resolvedFormId === 'addPerformanceForm') {
                console.log('🏆 Performance form submitted successfully, refreshing content...');
                showPage('performance');
                console.log('🔄 Waiting for page transition, then calling updatePerformanceContent()...');
                setTimeout(async () => {
                    console.log('🔄 Now calling updatePerformanceContent()...');
                    await updatePerformanceContent();
                    console.log('✅ Performance content update completed');
                }, 100);
            } else if (resolvedFormId === 'editPerformanceForm') {
                console.log('✏️ Performance edit form submitted successfully, applying optimistic row update + refresh...');
                try {
                    const recId = document.getElementById('editPerformanceId')?.value;
                    if (recId) {
                        const row = document.querySelector(`tr[data-record-id="${recId}"]`);
                        if (row) {
                            const cells = row.querySelectorAll('td');
                            if (cells.length >= 10) {
                                // data'daki alan adları addPerformanceForm ile uyumlu; backend tablo doldurma sırasına göre yazalım
                                cells[0].textContent = document.getElementById('editPerformanceDate')?.value || cells[0].textContent;
                                // Personel adı server’dan geliyor; hızlı yama yapmıyoruz
                                cells[2].textContent = document.getElementById('editPerformanceMemberCount')?.value || '0';
                                cells[3].textContent = document.getElementById('editPerformanceWhatsappCount')?.value || '0';
                                cells[4].textContent = document.getElementById('editPerformanceDeviceCount')?.value || '0';
                                cells[5].textContent = document.getElementById('editPerformanceUnansweredCount')?.value || '0';
                                cells[6].textContent = document.getElementById('editPerformanceKnowledgeDuelResult')?.value || '0';
                                const rp = (document.getElementById('editPerformanceRewardPenalty')?.value || '').toString().toLowerCase();
                                cells[7].textContent = rp === 'odul' ? 'ödül' : '-';
                                cells[8].textContent = rp === 'ceza' ? 'ceza' : '-';
                                row.classList.add('highlight-updated');
                                setTimeout(() => row.classList.remove('highlight-updated'), 2000);
                            }
                        }
                    }
                } catch (e) { console.warn('Optimistic perf row update skipped:', e); }
                showPage('performance');
                updatePerformanceContent();
                setTimeout(() => { updatePerformanceContent().catch(()=>{}); }, 800);
            } else if (resolvedFormId === 'addTrainingFeedbackForm') {
                console.log('🎓 Training feedback form submitted successfully, refreshing content...');
                showPage('training-feedback');
                console.log('🔄 Waiting for page transition, then calling updateTrainingFeedbackContent()...');
                setTimeout(async () => {
                    console.log('🔄 Now calling updateTrainingFeedbackContent()...');
                    await updateTrainingFeedbackContent();
                    console.log('✅ Training feedback content update completed');
                }, 100);
            } else if (resolvedFormId === 'editTrainingFeedbackForm') {
                console.log('✏️ Training feedback edit form submitted successfully, applying optimistic row update + refresh...');
                try {
                    const recId = document.getElementById('editTrainingFeedbackId')?.value;
                    if (recId) {
                        const row = document.querySelector(`tr[data-record-id="${recId}"]`);
                        if (row) {
                            const cells = row.querySelectorAll('td');
                            // New column order: 0 Tarih, 1 Personel, 2 Genel Eğitim Adedi, 3 Genel Eğitim Konusu, 4 Birebir Eğitimler Adedi, 5 Birebir Eğitimler Konusu, 6 Geribildirimler Adedi, 7 Geribildirimler Konusu, 8 İşlemler
                            if (cells.length >= 9) {
                                cells[0].textContent = document.getElementById('edit_training_feedback_date')?.value || cells[0].textContent;
                                // cells[1] personel adı; sunucudan gelir
                                cells[2].textContent = document.getElementById('edit_general_training_count')?.value || '0';
                                cells[3].textContent = document.getElementById('edit_general_training_subject')?.value || cells[3].textContent;
                                cells[4].textContent = document.getElementById('edit_one_on_one_training_count')?.value || '0';
                                cells[5].textContent = document.getElementById('edit_one_on_one_training_subject')?.value || cells[5].textContent;
                                cells[6].textContent = document.getElementById('edit_feedback_count')?.value || '0';
                                cells[7].textContent = document.getElementById('edit_feedback_subject')?.value || cells[7].textContent;
                                row.classList.add('highlight-updated');
                                setTimeout(() => row.classList.remove('highlight-updated'), 2000);
                            }
                        }
                    }
                } catch (e) { console.warn('Optimistic TF row update skipped:', e); }
                showPage('training-feedback');
                updateTrainingFeedbackContent();
                setTimeout(() => { updateTrainingFeedbackContent().catch(()=>{}); }, 800);
            } else if (resolvedFormId === 'addRecordForm') {
                console.log('📋 Record form submitted successfully, refreshing content...');
                showPage('records');
                console.log('🔄 Waiting for page transition, then calling updateRecordsContent()...');
                setTimeout(async () => {
                    console.log('🔄 Now calling updateRecordsContent()...');
                    await updateRecordsContent();
                    console.log('✅ Records content update completed');
                }, 100);
            } else if (resolvedFormId === 'editDailyRecordForm') {
                console.log('✏️ Daily record edit submitted, applying optimistic update + refresh...');
                try {
                    const recId = document.getElementById('editDailyRecordId')?.value;
                    if (recId) {
                        const row = document.querySelector(`tr[data-record-id="${recId}"]`);
                        if (row) {
                            const cells = row.querySelectorAll('td');
                            if (cells.length >= 6) {
                                cells[0].textContent = document.getElementById('editDailyRecordDate')?.value || cells[0].textContent;
                                // cells[1] personnel name is server-derived; keep as is for optimistic
                                cells[2].textContent = document.getElementById('editDailyRecordCallNumber')?.value || cells[2].textContent;
                                cells[3].textContent = document.getElementById('editDailyRecordScore')?.value || cells[3].textContent;
                                cells[4].textContent = document.getElementById('editDailyRecordNotes')?.value || cells[4].textContent;
                                row.classList.add('highlight-updated');
                                setTimeout(() => row.classList.remove('highlight-updated'), 2000);
                            }
                        }
                    }
                } catch (e) { console.warn('Optimistic daily record update skipped:', e); }
                showPage('records');
                updateRecordsContent();
                setTimeout(() => { updateRecordsContent().catch(()=>{}); }, 800);
            } else if (resolvedFormId === 'setTargetsForm') {
                console.log('🎯 Target saved, refreshing Targets page...');
                showPage('targets');
                setTimeout(() => { updateTargetsContent().catch(()=>{}); }, 100);
                // For dashboard "Hedef Süreci": show only count of personnel added in this op
                try {
                    const pidRaw = (data.target_personnel_id || '').toString().trim();
                    if (pidRaw) {
                        const uniqueIds = pidRaw.split(',').map(s => s.trim()).filter(Boolean);
                        dashboardState.lastTargetsAddedCount = uniqueIds.length;
                    } else {
                        dashboardState.lastTargetsAddedCount = 1; // default single add
                    }
                } catch { dashboardState.lastTargetsAddedCount = 1; }
                try { updateDashboardContent(); } catch {}
            } else if (resolvedFormId === 'afterHoursForm') {
                console.log('🕒 After Hours saved, refreshing Mesai Sonrası page...');
                // Clear hidden edit id to switch back to add-mode next time
                try { const hid = document.getElementById('afterHoursEditId'); if (hid) hid.value = ''; } catch {}
                showPage('analytics');
                setTimeout(() => { updateAfterHoursContent && updateAfterHoursContent(); }, 120);
            } else if (resolvedFormId === 'attendanceOverrideForm') {
                console.log('🗓️ Attendance override saved, refreshing Puantaj...');
                showPage('attendance');
                setTimeout(() => { updateAttendanceContent && updateAttendanceContent(); }, 120);
            } else if (resolvedFormId === 'warningCutForm') {
                console.log('⚠️ Warning/Cut saved, refreshing Puantaj...');
                showPage('attendance');
                setTimeout(() => { updateAttendanceContent && updateAttendanceContent(); }, 120);
            }
        } else {
            console.log('❌ Response not OK. Status:', response.status);
            const error = await response.json();
            console.error('❌ Error details:', error);
            
            // Show error notification
            showNotification(
                error.detail || 'İşlem başarısız oldu',
                'error',
                errorMessage
            );
        }
    } catch (error) {
        console.error('❌ Network error:', error);
        
        // Show network error notification
        showNotification(
            'Sunucu ile bağlantı kurulamadı. Lütfen backend sunucusunun çalıştığından emin olun.',
            'error',
            'Bağlantı Hatası'
        );
    }
}

function handleGlobalClick(event) {
    const target = event.target;
    console.log('🖱️ GLOBAL CLICK DETECTED:', {
        tag: target.tagName, 
        id: target.id, 
        classList: Array.from(target.classList),
        text: target.textContent.substring(0, 30),
        timestamp: new Date().toLocaleTimeString()
    });
    
    // 1. NAVIGATION - nav-link class'ına sahip elementler veya parent'ı nav-link olan elementler
    let navElement = target;
    
    // Target nav-link değilse, parent'larını kontrol et
    if (!target.classList.contains('nav-link')) {
        navElement = target.closest('.nav-link');
    }
    
    if (navElement && navElement.classList.contains('nav-link')) {
        event.preventDefault();
        console.log('🔗 NAV CLICKED:', navElement.id);
        
        const pageMap = {
            'navDashboard': 'dashboard',
            'navPersonnel': 'personnel',
            'navPerformance': 'performance',
            'navTrainingFeedback': 'training-feedback',
            'navRecords': 'records', 
            'navTargets': 'targets',
            'navAnalytics': 'analytics',
            'navAttendance': 'attendance',
            'navUsers': 'users',
            'navLogin': 'login'
        };
        
        const page = pageMap[navElement.id];
        if (navElement.id === 'navLogin') {
            if (isLoggedIn()) {
                setAuthToken('');
                try { window.__authUser = null; } catch {}
                showNotification('Çıkış yapıldı', 'info', 'Bilgi');
                showPage('login');
                return;
            }
        }
        if (page) {
            if (!isLoggedIn() && page !== 'login') {
                showNotification('Lütfen önce giriş yapın', 'warning', 'Uyarı');
                showPage('login');
                return;
            }
            console.log(`🔗 Navigating to: ${page}`);
            showPage(page);
        }
        return;
    }
    
    // 2. MODAL KAPATMA
    if (target.classList.contains('modal-close') || 
       (target.classList.contains('btn-secondary') && target.textContent.includes('İptal'))) {
        event.preventDefault();
        console.log('❌ Closing modal');
        closeModals();
        return;
    }
    
    // 3. MODAL AÇMA BUTONLARI - USE BUTTON STATE MANAGEMENT
    if (target.id === 'addPersonnelBtn') {
        event.preventDefault();
        console.log('👤 ADD PERSONNEL BUTTON CLICKED!');
        handleButtonClick(target, () => openModal('addPersonnelModal'), 500);
        return;
    }
    
    if (target.id === 'exportPersonnelBtn') {
        event.preventDefault();
        console.log('📊 EXPORT PERSONNEL BUTTON CLICKED!');
        handleButtonClick(target, () => exportPersonnelToExcel(), 2000);
        return;
    }
    
    if (target.id === 'exportPerformanceExcelBtn') {
        event.preventDefault();
        console.log('📊 EXPORT PERFORMANCE EXCEL BUTTON CLICKED!');
        handleButtonClick(target, () => exportPerformanceToExcel(), 2000);
        return;
    }
    
    if (target.id === 'addRecordBtn') {
        event.preventDefault();
        console.log('📋 ADD RECORD BUTTON CLICKED!');
        handleButtonClick(target, () => openModal('addRecordModal'), 500);
        return;
    }
    
    if (target.id === 'addPerformanceBtn') {
        event.preventDefault();
        console.log('🏆 ADD PERFORMANCE BUTTON CLICKED!');
        handleButtonClick(target, () => openModal('addPerformanceModal'), 500);
        return;
    }
    
    if (target.id === 'addTrainingFeedbackBtn') {
        event.preventDefault();
        console.log('🎓 ADD TRAINING FEEDBACK BUTTON CLICKED!');
        handleButtonClick(target, () => openModal('addTrainingFeedbackModal'), 500);
        return;
    }
    
    if (target.id === 'exportExcelBtn') {
        event.preventDefault();
        console.log('📊 RECORDS EXCEL EXPORT BUTTON CLICKED!');
        handleButtonClick(target, () => exportRecordsExcel(), 1500);
        return;
    }
    if (target.id === 'exportExcelBtnAnalytics') {
        event.preventDefault();
        console.log('📊 ANALYTICS EXCEL EXPORT BUTTON CLICKED!');
        handleButtonClick(target, () => openModal('exportExcelModal'), 500);
        return;
    }
    
    if (target.id === 'exportTrainingFeedbackExcelBtn') {
        event.preventDefault();
        console.log('📊 TRAINING FEEDBACK EXCEL EXPORT BUTTON CLICKED!');
        handleButtonClick(target, () => exportTrainingFeedbackExcel(), 2000);
        return;
    }

    // MESAI SONRASI (After Hours) BUTTONS
    if (target.id === 'addAfterHoursBtn') {
        event.preventDefault();
        console.log('🕒 ADD AFTER HOURS BUTTON CLICKED!');
        handleButtonClick(target, async () => {
            await loadPersonnelOptions('after_hours_personnel_id');
            // Default date to today
            const dt = document.getElementById('after_hours_date');
            if (dt && !dt.value) dt.value = new Date().toISOString().split('T')[0];
            // Clear hidden edit id to ensure add-mode
            const hid = document.getElementById('afterHoursEditId');
            if (hid) hid.value = '';
            openModal('afterHoursModal');
        }, 500);
        return;
    }
    if (target.id === 'applyAfterHoursFilter') {
        event.preventDefault();
        console.log('📅 APPLY AFTER HOURS FILTER CLICKED!');
        handleButtonClick(target, () => updateAfterHoursContent(), 800);
        return;
    }
    if (target.id === 'clearAfterHoursFilter') {
        event.preventDefault();
        console.log('🗑️ CLEAR AFTER HOURS FILTER CLICKED!');
        handleButtonClick(target, () => clearAfterHoursFilter(), 500);
        return;
    }
    if (target.id === 'exportAfterHoursExcelBtn') {
        event.preventDefault();
        console.log('📊 AFTER HOURS EXCEL EXPORT CLICKED!');
        handleButtonClick(target, () => exportAfterHoursExcel(), 1500);
        return;
    }
    // ATTENDANCE (Puantaj) buttons
    if (target.id === 'applyAttendanceFilter') {
        event.preventDefault();
        console.log('📅 APPLY ATTENDANCE FILTER CLICKED!');
        handleButtonClick(target, () => updateAttendanceContent(), 800);
        return;
    }
    if (target.id === 'clearAttendanceFilter') {
        event.preventDefault();
        console.log('🗑️ CLEAR ATTENDANCE FILTER CLICKED!');
        handleButtonClick(target, () => { clearAttendanceFilterUI(); updateAttendanceContent(); }, 500);
        return;
    }
    if (target.id === 'addAttendanceOverrideBtn') {
        event.preventDefault();
        console.log('➕ ADD ATTENDANCE OVERRIDE CLICKED!');
        handleButtonClick(target, async () => {
            // Reset form and hide delete button
            const form = document.getElementById('attendanceOverrideForm');
            if (form && typeof form.reset === 'function') form.reset();
            const hid = document.getElementById('attendance_override_edit_id');
            if (hid) hid.value = '';
            const delBtn = document.getElementById('deleteAttendanceOverrideBtn');
            if (delBtn) delBtn.style.display = 'none';
            // Preselect date range end date
            const end = document.getElementById('attendanceEndDate')?.value || new Date().toISOString().split('T')[0];
            const dateInput = document.getElementById('attendance_override_date');
            if (dateInput) dateInput.value = end;
            // Load personnel options
            await loadPersonnelOptions('attendance_override_personnel_id');
            openModal('attendanceOverrideModal');
        }, 500);
        return;
    }
    if (target.id === 'exportWarningsCutsBtn') {
        event.preventDefault();
        console.log('📊 WARNINGS/CUTS EXPORT CLICKED!');
        handleButtonClick(target, () => exportWarningsCuts(), 1500);
        return;
    }
    // USERS PAGE BUTTONS
    if (target.id === 'addUserBtn') {
        event.preventDefault();
        console.log('👤 ADD USER CLICKED!');
        handleButtonClick(target, () => openAddUserModal(), 400);
        return;
    }
    const userEditBtn = target.closest && target.closest('.user-edit-btn');
    if (userEditBtn) {
        event.preventDefault();
        const uid = Number(userEditBtn.getAttribute('data-user-id'));
        handleButtonClick(userEditBtn, () => openEditUserModal(uid), 400);
        return;
    }
    const userDeleteBtn = target.closest && target.closest('.user-delete-btn');
    if (userDeleteBtn) {
        event.preventDefault();
        const uid = Number(userDeleteBtn.getAttribute('data-user-id'));
        handleButtonClick(userDeleteBtn, async () => { try { await deleteUser(uid); await updateUsersContent(); showNotification('Kullanıcı silindi', 'success', 'Başarılı'); } catch(err){ showNotification(err.message||'Silme hatası', 'error', 'Hata'); } }, 800);
        return;
    }
    // UYARI - KESİNTİ add button
    if (target.id === 'addWarningCutBtn') {
        event.preventDefault();
        console.log('➕ ADD WARNING/CUT CLICKED!');
        handleButtonClick(target, async () => {
            // Reset form
            const form = document.getElementById('warningCutForm');
            if (form && typeof form.reset === 'function') form.reset();
            // Clear edit id to enter add mode
            try { const hid = document.getElementById('warning_cut_edit_id'); if (hid) hid.value = ''; } catch {}
            // Preselect date range end date
            const end = document.getElementById('attendanceEndDate')?.value || new Date().toISOString().split('T')[0];
            const dateInput = document.getElementById('warning_cut_date');
            if (dateInput) dateInput.value = end;
            // Load personnel options
            await loadPersonnelOptions('warning_cut_personnel_id');
            openModal('warningCutModal');
        }, 500);
        return;
    }
    
    if (target.id === 'setTargetsBtn' || target.id === 'setTargetsBtn2') {
        event.preventDefault();
        console.log('🎯 SET TARGETS BUTTON CLICKED!');
        // Clear add form state
        try {
            const form = document.getElementById('setTargetsForm');
            if (form && typeof form.reset === 'function') form.reset();
            const hid = document.getElementById('editTargetId');
            if (hid) hid.value = '';
        } catch {}
        handleButtonClick(target, () => openModal('setTargetsModal'), 500);
        return;
    }
    if (target.id === 'refreshTargetsBtn') {
        event.preventDefault();
        console.log('🔄 REFRESH TARGETS BUTTON CLICKED!');
        handleButtonClick(target, () => updateTargetsContent(), 800);
        return;
    }
    
    if (target.id === 'refreshBtn') {
        event.preventDefault();
        console.log('🔄 REFRESH BUTTON CLICKED!');
        handleButtonClick(target, () => location.reload(), 1000);
        return;
    }
    
    // PERFORMANS TARIH FİLTRE BUTONLARI
    if (target.id === 'applyPerformanceDateFilter') {
        event.preventDefault();
        console.log('📅 APPLY PERFORMANCE DATE FILTER CLICKED!');
        handleButtonClick(target, () => applyPerformanceDateFilter(), 1000);
        return;
    }
    
    if (target.id === 'clearPerformanceDateFilter') {
        event.preventDefault();
        console.log('🗑️ CLEAR PERFORMANCE DATE FILTER CLICKED!');
        handleButtonClick(target, () => clearPerformanceDateFilter(), 500);
        return;
    }
    
    // KAYITLAR (ÇAĞRI PUANLARI) TARIH FİLTRE BUTONLARI
    if (target.id === 'applyRecordsDateFilter') {
        event.preventDefault();
        console.log('📅 APPLY RECORDS DATE FILTER CLICKED!');
        handleButtonClick(target, () => applyRecordsDateFilter(), 1000);
        return;
    }
    
    if (target.id === 'clearRecordsDateFilter') {
        event.preventDefault();
        console.log('🗑️ CLEAR RECORDS DATE FILTER CLICKED!');
        handleButtonClick(target, () => clearRecordsDateFilter(), 500);
        return;
    }

    // EĞİTİM-GERİBİLDİRİM TARİH FİLTRE BUTONLARI
    if (target.id === 'applyTrainingFeedbackFilter') {
        event.preventDefault();
        console.log('📅 APPLY TRAINING-FEEDBACK DATE FILTER CLICKED!');
        handleButtonClick(target, () => applyTrainingFeedbackFilter(), 1000);
        return;
    }

    if (target.id === 'clearTrainingFeedbackFilter') {
        event.preventDefault();
        console.log('🗑️ CLEAR TRAINING-FEEDBACK DATE FILTER CLICKED!');
        handleButtonClick(target, () => clearTrainingFeedbackFilter(), 500);
        return;
    }
    
    // EDIT VE DELETE BUTONLARI
    const editBtnEl = target.closest && target.closest('.edit-btn');
    if (editBtnEl) {
        event.preventDefault();
        const recordId = parseInt(editBtnEl.getAttribute('data-record-id'));
        console.log(`✏️ EDIT BUTTON CLICKED! Record ID: ${recordId}`);
        
        handleButtonClick(editBtnEl, () => {
            const currentPage = document.querySelector('.page.active');
            if (currentPage) {
                const pageId = currentPage.id;
                
                if (pageId === 'training-feedback-page') {
                    editTrainingFeedbackRecord(recordId);
                } else if (pageId === 'personnel-page') {
                    editPersonnelRecord(recordId);
                } else if (pageId === 'performance-page') {
                    editPerformanceRecord(recordId);
                } else if (pageId === 'records-page') {
                    editDailyRecord(recordId);
                } else if (pageId === 'analytics-page') {
                    editAfterHoursRecord(recordId);
                }
            }
        }, 500);
        return;
    }
    
    const deleteBtnEl = target.closest && target.closest('.delete-btn');
    if (deleteBtnEl) {
        event.preventDefault();
        const recordId = parseInt(deleteBtnEl.getAttribute('data-record-id'));
        console.log(`🗑️ DELETE BUTTON CLICKED! Record ID: ${recordId}`);
        
        handleButtonClick(deleteBtnEl, () => {
            const currentPage = document.querySelector('.page.active');
            if (currentPage) {
                const pageId = currentPage.id;
                
                if (pageId === 'training-feedback-page') {
                    deleteTrainingFeedbackRecord(recordId);
                } else if (pageId === 'personnel-page') {
                    deletePersonnelRecord(recordId);
                } else if (pageId === 'performance-page') {
                    deletePerformanceRecord(recordId);
                } else if (pageId === 'records-page') {
                    deleteDailyRecord(recordId);
                } else if (pageId === 'analytics-page') {
                    deleteAfterHoursRecord(recordId);
                }
            }
        }, 1000);
        return;
    }

    // === TARGETS: Edit/Delete buttons (delegated) ===
    const targetEditBtn = target.closest && target.closest('.target-edit-btn');
    if (targetEditBtn) {
        event.preventDefault();
        const tid = parseInt(targetEditBtn.getAttribute('data-target-id'));
        console.log(`✏️ TARGET EDIT CLICKED! Target ID: ${tid}`);
        handleButtonClick(targetEditBtn, () => editTarget(tid), 500);
        return;
    }
    const targetDeleteBtn = target.closest && target.closest('.target-delete-btn');
    if (targetDeleteBtn) {
        event.preventDefault();
        const tid = parseInt(targetDeleteBtn.getAttribute('data-target-id'));
        console.log(`🗑️ TARGET DELETE CLICKED! Target ID: ${tid}`);
        handleButtonClick(targetDeleteBtn, () => deleteTarget(tid), 800);
        return;
    }
    
    // 4. MODAL OVERLAY KAPATMA
    if (event.target.id === 'modalOverlay') {
        closeModals();
        return;
    }
    
    // 5. FORM SUBMIT BUTONLARI - SADECE SUBMIT TYPE BUTONLAR
    if (target.tagName === 'BUTTON' && 
        (target.type === 'submit' || target.getAttribute('type') === 'submit') &&
        target.textContent && !target.textContent.includes('İptal')) {

        // Form'u bul - önce form attribute'u, sonra closest
        let form = target.form;
        if (!form && target.getAttribute('form')) {
            form = document.getElementById(target.getAttribute('form'));
        }
        if (!form) {
            form = target.closest('form');
        }

    if (form) {
            const fid = typeof form.getAttribute === 'function' ? form.getAttribute('id') : form.id;
            // Login formu kendi onsubmit ile çalışsın; global handleFormSubmitInline devre dışı
            if (fid === 'loginForm' || target.id === 'loginSubmitBtn') {
        console.log('🔓 Skipping global submit button intercept for login');
        try { event.preventDefault(); } catch {}
        try { if (typeof loginAttempt === 'function') loginAttempt(); } catch {}
        return;
            }
            // Users: özel-case (add/edit)
            if (fid === 'addUserForm') {
                event.preventDefault();
                console.log('👤 SUBMIT addUserForm (button intercept)');
                handleButtonClick(target, async () => {
                    await submitAddUser(form);
                    closeModals();
                    showNotification('Kullanıcı oluşturuldu', 'success', 'Başarılı');
                    await updateUsersContent();
                }, 1200);
                return;
            }
            if (fid === 'editUserForm') {
                event.preventDefault();
                console.log('✏️ SUBMIT editUserForm (button intercept)');
                handleButtonClick(target, async () => {
                    await submitEditUser(form);
                    closeModals();
                    showNotification('Kullanıcı güncellendi', 'success', 'Başarılı');
                    await updateUsersContent();
                }, 1200);
                return;
            }
            // Diğer tüm formlar için default'u engelle ve inline handler'ı çalıştır
            event.preventDefault();
            console.log('📝 FORM SUBMIT BUTTON CLICKED:', target.textContent, form.id);
            handleButtonClick(target, () => handleFormSubmitInline(form), 2000);
        } else {
            // Form bulunamadı; yine de default submit'i engelleme (beklenen davranış olmayabilir)
            console.log('❌ Form not found for submit button:', target.textContent);
        }
        return;
    }

    // Attendance grid cell click -> open edit if override exists
    const attendanceCell = target.closest && target.closest('#attendanceTableBody td');
    if (attendanceCell) {
        const rowEl = attendanceCell.closest('tr');
        const table = attendanceCell.closest('table');
        const headRow = document.querySelector('#attendanceTableHead tr');
        if (rowEl && table && table.id && table.id.includes('attendance') && headRow) {
            const cellIndex = Array.from(rowEl.children).indexOf(attendanceCell);
            if (cellIndex <= 0) return; // ignore name column
            const dateText = headRow.children[cellIndex]?.getAttribute('title');
            if (!dateText || !/\d{4}-\d{2}-\d{2}/.test(dateText)) return;
            const personId = rowEl.getAttribute('data-personnel-id');
            if (!personId) return;
            (async () => {
                await loadPersonnelOptions('attendance_override_personnel_id');
                const form = document.getElementById('attendanceOverrideForm');
                if (form && typeof form.reset === 'function') form.reset();
                const delBtn = document.getElementById('deleteAttendanceOverrideBtn');
                const editId = attendanceCell.getAttribute('data-override-id');
                if (editId) {
                    // Prefill with existing override
                    const ovMap = window.__attendanceOverridesMap || new Map();
                    const key = `${personId}|${dateText}`;
                    const ov = ovMap.get ? ovMap.get(key) : null;
                    document.getElementById('attendance_override_edit_id').value = String(editId);
                    document.getElementById('attendance_override_personnel_id').value = String(personId);
                    document.getElementById('attendance_override_date').value = dateText;
                    if (ov) {
                        const isWeekly = (ov.leave_type && String(ov.leave_type).toLowerCase() === 'weekly') || (ov.period && String(ov.period).toLowerCase() === 'weekly');
                        document.getElementById('attendance_override_value').value = isWeekly ? '1-week' : String(ov.value);
                        if (ov.notes) document.getElementById('attendance_override_notes').value = ov.notes;
                    }
                    if (delBtn) delBtn.style.display = '';
                } else {
                    // New add
                    document.getElementById('attendance_override_edit_id').value = '';
                    document.getElementById('attendance_override_personnel_id').value = String(personId);
                    document.getElementById('attendance_override_date').value = dateText;
                    document.getElementById('attendance_override_value').value = '1';
                    if (delBtn) delBtn.style.display = 'none';
                }
                openModal('attendanceOverrideModal');
            })();
            return;
        }
    }
    
    // 6. MODAL CLOSE BUTTONS
    if (target.classList.contains('modal-close') || target.textContent === 'İptal') {
        event.preventDefault();
        console.log('❌ MODAL CLOSE BUTTON CLICKED!');
        closeModals();
        return;
    }
    
    // 7. MODAL OVERLAY CLICK (backdrop click to close)
    if (target.classList.contains('modal-overlay')) {
        event.preventDefault();
        console.log('🎭 MODAL OVERLAY CLICKED - CLOSING MODAL!');
        closeModals();
        return;
    }
}

// Safe notification function
function showNotification(message, type = 'success', title = '') {
    if (toastManager && toastManager.container) {
        toastManager.show(message, type, title);
    } else {
        // Fallback to console if toast system fails
        const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : '⚠️';
        console.log(`${icon} ${title ? title + ': ' : ''}${message}`);
    }
}

// DOM yüklendiğinde çalış
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM loaded - VERSION 6.28-TOAST-FIX');
    console.log('🔧 Setting up event listeners...');
    
    // Initialize toast manager with retry
    function initToastManager() {
        const container = document.getElementById('toastContainer');
        if (container) {
            toastManager = new ToastManager();
            console.log('✅ Toast Manager initialized successfully');
            return true;
        } else {
            console.warn('⚠️ Toast container not found, retrying...');
            return false;
        }
    }
    
    // Try to initialize immediately, with fallback
    if (!initToastManager()) {
        setTimeout(() => {
            if (!initToastManager()) {
                console.error('❌ Failed to initialize toast manager');
            }
        }, 500);
    }
    
    // CRITICAL: Test click listener immediately
    console.log('🧪 Testing click listener setup...');
    
    // Event delegation setup
    setupEventListeners();
    
    // Show login page if not authenticated
    showPage(isLoggedIn() ? 'dashboard' : 'login');
    // Hydrate current user from token then update nav visibility
    (async () => { if (isLoggedIn()) { try { await hydrateAuthUser(); } catch {} } updateAuthNavUI(); })();
    // Ensure dashboard KPIs render once DOM is ready
    try { updateDashboardContent(); } catch {}
    
    console.log('✅ App initialized successfully - v6.22');
    
    // CRITICAL: Add immediate test
    setTimeout(() => {
        console.log('🔍 POST-INIT CHECK: Event listeners should be active now');
        const testClicks = document.querySelectorAll('button, .btn, .nav-link');
        console.log(`📊 Found ${testClicks.length} clickable elements`);
        
        // Add debug alert to ANY click on page
        document.body.addEventListener('click', function(e) {
            console.log('🎯 BODY CLICK DETECTED:', e.target.tagName, e.target.className, e.target.id);
        });
        
        console.log('🎯 Body click listener added successfully!');
    }, 1000);
});

function setupEventListeners() {
    console.log('🔧 Setting up event listeners - DETAILED VERSION');
    console.log('📍 Document ready state:', document.readyState);
    console.log('📍 Document body exists:', !!document.body);
    
    // PREVENT DUPLICATE EVENT LISTENERS
    if (globalEventListenerAttached) {
        console.log('⚠️ Event listeners already attached, skipping...');
        return;
    }
    
    // TEST: Basit bir test butonu event'i ekle
    console.log('🧪 Testing if click events work at all...');
    
    // TEK GLOBAL CLICK LISTENER - SADECE BİR KEZ EKLEME
    if (!globalClickHandlerAttached) {
        document.addEventListener('click', handleGlobalClick, { passive: false });
        // Ayrıca submit event'ini capture aşamasında yakalayarak gerçek FORM'u iletelim
        document.addEventListener('submit', function(e){
            try {
                const f = e.target;
                const formId = f && (typeof f.getAttribute === 'function' ? f.getAttribute('id') : f.id);
                console.log('🧲 GLOBAL SUBMIT CAPTURED:', formId);
                // Login formu özel onsubmit ile yönetiliyor; global interceptor atlanmalı
                if (formId === 'loginForm' || (typeof f.closest === 'function' && f.closest('#login-page'))) {
                    console.log('🔓 Skipping global submit intercept for login form');
                    return; // preventDefault çağırmadan bırak
                }
                if (formId === 'addUserForm') {
                    e.preventDefault();
                    (async ()=>{ try { await submitAddUser(f); closeModals(); showNotification('Kullanıcı oluşturuldu', 'success', 'Başarılı'); await updateUsersContent(); } catch(err){ showNotification(err.message||'Hata', 'error', 'Hata'); } })();
                    return;
                }
                if (formId === 'editUserForm') {
                    e.preventDefault();
                    (async ()=>{ try { await submitEditUser(f); closeModals(); showNotification('Kullanıcı güncellendi', 'success', 'Başarılı'); await updateUsersContent(); } catch(err){ showNotification(err.message||'Hata', 'error', 'Hata'); } })();
                    return;
                }
                if (f && f.tagName === 'FORM') {
                    e.preventDefault();
                    handleFormSubmitInline(f);
                }
            } catch (err) {
                console.error('❌ Global submit handler error:', err);
            }
        }, true);
        globalClickHandlerAttached = true;
        console.log('✅ Global click & submit handlers attached ONCE');
    }
    // Dropdown change listener'ları ekle - SADECE BİR KEZ
    if (!document.body.dataset.changeListenerAttached) {
        document.addEventListener('change', function(event) {
            const target = event.target;
            
            // Performans personel filtresi
            if (target.id === 'performancePersonnelFilter') {
                console.log('👤 Performance personnel filter changed:', target.value);
                // Otomatik filtrele
                applyPerformanceDateFilter();
            }
            // Records personel filtresi
            if (target.id === 'personnelFilter') {
                console.log('👤 Records personnel filter changed:', target.value);
                applyRecordsDateFilter();
            }
        });
        document.body.dataset.changeListenerAttached = 'true';
        console.log('✅ Change listeners attached ONCE');
    }
    
    globalEventListenerAttached = true;
    console.log('✅ Event listeners set up successfully - NO DUPLICATES');
}

async function showPage(pageId) {
    console.log(`📄 showPage called with: ${pageId}`); 
    // Guard: Users page is admin-only
    try {
        if (pageId === 'users') {
            const u = (typeof window !== 'undefined' && window.__authUser) ? window.__authUser : null;
            const isAdmin = !!(u && u.role === 'admin');
            if (!isLoggedIn() || !isAdmin) {
                showNotification('Bu sayfa yalnızca admin kullanıcılar içindir', 'warning', 'Uyarı');
                pageId = 'dashboard';
            }
        }
    } catch {}
    
    // 1. Tüm sayfaları gizle
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // 2. Hedef sayfayı göster
    const targetPageId = pageId + '-page';
    console.log(`🎯 Looking for page element with ID: ${targetPageId}`);
    const targetPage = document.getElementById(targetPageId);
    console.log(`🔍 Found page element:`, targetPage);
    
    if (targetPage) {
        targetPage.classList.add('active');
        console.log(`✅ Page shown: ${targetPageId}`);
    } else {
        console.error(`❌ Page not found: ${targetPageId}`);
        console.log('Available page elements:', Array.from(document.querySelectorAll('.page')).map(p => p.id));
        return;
    }
    
    // 3. Navigation güncelle
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    const navIdMap = {
        'dashboard': 'navDashboard',
        'personnel': 'navPersonnel',
        'performance': 'navPerformance',
        'training-feedback': 'navTrainingFeedback',
        'records': 'navRecords', 
        'targets': 'navTargets',
        'analytics': 'navAnalytics',
        'attendance': 'navAttendance',
    'users': 'navUsers',
        'login': 'navLogin'
    };
    
    const navElement = document.getElementById(navIdMap[pageId]);
    if (navElement) {
        navElement.classList.add('active');
        console.log(`✅ Nav updated: ${navIdMap[pageId]}`);
    }
    
    // 4. Sayfa content'i güncelle
    await updatePageContent(pageId);
    updateAuthNavUI();

    // If dashboard, refresh KPIs again after content switch
    if (pageId === 'dashboard') {
        try { updateDashboardContent(); } catch {}
    }
}

async function updatePageContent(pageId) {
    console.log(`🔄 Updating content for: ${pageId}`);
    
    switch(pageId) {
        case 'dashboard':
            updateDashboardContent();
            break;
        case 'login':
            initLoginPage();
            break;
        case 'personnel':
            await updatePersonnelContent();
            break;
        case 'performance':
            await updatePerformanceContent();
            break;
        case 'training-feedback':
            await updateTrainingFeedbackContent();
            break;
        case 'records':
            await updateRecordsContent();
            break;
        case 'analytics':
            await updateAfterHoursContent();
            break;
        case 'attendance':
            await updateAttendanceContent();
            break;
        case 'users':
            await updateUsersContent();
            break;
        case 'targets':
            updateTargetsContent();
            break;
    }
}

function updateAuthNavUI(){
    try {
        const logged = isLoggedIn();
        const ids = ['navDashboard','navPersonnel','navPerformance','navTrainingFeedback','navRecords','navTargets','navAnalytics','navAttendance'];
        ids.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.pointerEvents = logged ? '' : 'auto';
            if (el) el.classList.toggle('disabled', !logged);
        });
        const loginEl = document.getElementById('navLogin');
        if (loginEl) loginEl.textContent = logged ? 'Çıkış' : 'Giriş';
        // Role-based visibility for Users nav
        try {
            const u = window.__authUser || null;
            const isAdmin = !!(u && (u.role === 'admin'));
            const usersNav = document.getElementById('navUsers');
            if (usersNav) usersNav.style.display = logged && isAdmin ? '' : 'none';
        } catch {}
    } catch {}
}

// === Login Page Logic ===
async function loginAttempt(){
    try {
        const submitBtn = document.getElementById('loginSubmitBtn');
        const u = (document.getElementById('login_username')?.value || '').toString().trim();
        const p = (document.getElementById('login_password')?.value || '').toString().trim();
        if (!u || !p) { showNotification('Kullanıcı adı ve şifre gerekli', 'warning', 'Uyarı'); return; }
        if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Giriş Yap ⏳'; }
        const resp = await fetch(`${API_BASE_URL}/api/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: u, password: p }) });
        if (!resp.ok) {
            const err = await resp.json().catch(()=>({}));
            showNotification(err.detail || 'Giriş başarısız', 'error', 'Hata');
            return;
        }
        const json = await resp.json();
    const token = json?.data?.token || '';
        if (!token) { showNotification('Token alınamadı', 'error', 'Hata'); return; }
    setAuthToken(token);
    try { window.__authUser = json?.data?.user || null; } catch {}
        showNotification('Giriş başarılı', 'success', 'Başarılı');
        showPage('dashboard');
    } catch (e) {
        showNotification('Sunucuya ulaşılamadı', 'error', 'Hata');
    } finally {
        const submitBtn = document.getElementById('loginSubmitBtn');
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Giriş Yap'; }
    }
}

// Populate current user info based on stored token
async function hydrateAuthUser(){
    try {
        if (!isLoggedIn()) { try { window.__authUser = null; } catch {} return null; }
        const resp = await fetch(`${API_BASE_URL}/api/auth/me`, { cache: 'no-store' });
        if (resp.ok) {
            const j = await resp.json().catch(()=>({}));
            try { window.__authUser = j?.data?.user || null; } catch { window.__authUser = null; }
            return window.__authUser;
        }
        if (resp.status === 401) {
            // Token invalid/expired
            setAuthToken('');
            try { window.__authUser = null; } catch {}
        }
    } catch {}
    return null;
}

// ===== Users Management =====
async function fetchUsers(){
    const resp = await fetch(`${API_BASE_URL}/api/auth/users`, { cache: 'no-store' });
    if (!resp.ok) throw new Error((await resp.json().catch(()=>({})))?.detail || 'Kullanıcı listesi alınamadı');
    const j = await resp.json();
    return Array.isArray(j?.data) ? j.data : [];
}

async function updateUsersContent(){
    const body = document.getElementById('usersTableBody');
    if (!body) return;
    // Role-based guard: only admins can view
    const me = (window.__authUser || null);
    if (!me || String(me.role) !== 'admin'){
        body.innerHTML = '<tr><td colspan="6" class="text-center">Bu sayfa için yetkiniz yok</td></tr>';
        return;
    }
    body.innerHTML = '<tr><td colspan="6" class="text-center">Yükleniyor...</td></tr>';
    try{
        const rows = await fetchUsers();
        if (!rows.length){
            body.innerHTML = '<tr><td colspan="6" class="text-center">Kullanıcı yok</td></tr>';
            return;
        }
        const html = rows.map(u => {
            const active = (Number(u.is_active) === 1 || u.is_active === true) ? 'Aktif' : 'Pasif';
            const created = (u.created_at || '').toString();
            return `<tr data-user-id="${u.id}">
                <td>${u.id}</td>
                <td>${u.username}</td>
                <td>${u.role}</td>
                <td>${active}</td>
                <td>${created.split('T')[0] || ''}</td>
                <td>
                    <button class="btn btn-sm btn-warning user-edit-btn" data-user-id="${u.id}"><i class="fas fa-edit"></i> Düzenle</button>
                    <button class="btn btn-sm btn-danger user-delete-btn" data-user-id="${u.id}"><i class="fas fa-trash"></i> Sil</button>
                </td>
            </tr>`;
        }).join('');
        body.innerHTML = html;
    } catch(e){
        console.error('Users load error', e);
        body.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Hata: ${e.message}</td></tr>`;
    }
}

function openAddUserModal(){
    try { const f = document.getElementById('addUserForm'); if (f && f.reset) f.reset(); } catch {}
    openModal('addUserModal');
}

function openEditUserModal(userId){
    const row = document.querySelector(`tr[data-user-id="${userId}"]`);
    if (!row) return;
    const cells = row.querySelectorAll('td');
    const username = cells[1]?.textContent?.trim() || '';
    const role = cells[2]?.textContent?.trim() || 'user';
    const isActive = (cells[3]?.textContent || '').includes('Aktif');
    document.getElementById('edit_user_id').value = String(userId);
    document.getElementById('edit_user_username').value = username;
    document.getElementById('edit_user_role').value = role;
    document.getElementById('edit_user_active').checked = isActive;
    const pw = document.getElementById('edit_user_password'); if (pw) pw.value = '';
    openModal('editUserModal');
}

async function submitAddUser(form){
    const u = form.username.value.trim();
    const p = form.password.value.trim();
    const role = form.role.value;
    const is_active = form.is_active.checked;
    const resp = await fetch(`${API_BASE_URL}/api/auth/users`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ username:u, password:p, role, is_active }) });
    if (!resp.ok) throw new Error((await resp.json().catch(()=>({})))?.detail || 'Kullanıcı oluşturulamadı');
    return await resp.json();
}

async function submitEditUser(form){
    const id = Number(document.getElementById('edit_user_id').value);
    const payload = { role: form.role.value, is_active: form.is_active.checked };
    const newPw = form.password.value.trim();
    if (newPw) payload.password = newPw;
    const resp = await fetch(`${API_BASE_URL}/api/auth/users/${id}`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    if (!resp.ok) throw new Error((await resp.json().catch(()=>({})))?.detail || 'Kullanıcı güncellenemedi');
    return await resp.json();
}

async function deleteUser(userId){
    const ok = confirm('Kullanıcıyı silmek istediğinize emin misiniz?');
    if (!ok) return;
    const resp = await fetch(`${API_BASE_URL}/api/auth/users/${userId}`, { method:'DELETE' });
    if (!resp.ok) throw new Error((await resp.json().catch(()=>({})))?.detail || 'Kullanıcı silinemedi');
    return await resp.json();
}

function initLoginPage(){
    try {
        const form = document.getElementById('loginForm');
        const logoutBtn = document.getElementById('logoutBtn');
        const submitBtn = document.getElementById('loginSubmitBtn');
        const status = document.getElementById('loginStatus');
        if (status) status.textContent = isLoggedIn() ? 'Giriş yapıldı' : '';
        if (logoutBtn) logoutBtn.style.display = isLoggedIn() ? '' : 'none';
        if (!form) return;
        form.onsubmit = (e) => { e.preventDefault(); loginAttempt(); };
        if (submitBtn && !submitBtn.dataset.loginClickBound) {
            submitBtn.addEventListener('click', (e) => { e.preventDefault(); loginAttempt(); });
            submitBtn.dataset.loginClickBound = 'true';
        }
        if (logoutBtn) {
            logoutBtn.onclick = () => {
                setAuthToken('');
                try { window.__authUser = null; } catch {}
                showNotification('Çıkış yapıldı', 'info', 'Bilgi');
                showPage('login');
            };
        }
    } catch {}
}

// ===== Puantaj (Attendance) =====
function clearAttendanceFilterUI() {
    const ids = ['attendanceStartDate','attendanceEndDate'];
    ids.forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
}

function formatDateISO(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

function buildDateRange(startStr, endStr) {
    const start = new Date(startStr + 'T00:00:00');
    const end = new Date(endStr + 'T00:00:00');
    const days = [];
    for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
        days.push(formatDateISO(d));
    }
    return days;
}

async function updateAttendanceContent() {
    const thead = document.getElementById('attendanceTableHead');
    const tbody = document.getElementById('attendanceTableBody');
    if (!thead || !tbody) return;

    // Resolve filters or defaults: current month to today
    const startInput = document.getElementById('attendanceStartDate');
    const endInput = document.getElementById('attendanceEndDate');
    let startDate = (startInput && startInput.value) || '';
    let endDate = (endInput && endInput.value) || '';
    const today = new Date();
    const todayStr = formatDateISO(today);
    if (!endDate) endDate = todayStr;
    if (!startDate) {
        const firstOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);
        startDate = formatDateISO(firstOfMonth);
    }
    // Guard invalid range
    if (startDate > endDate) {
        const tmp = startDate; startDate = endDate; endDate = tmp;
    }
    // Reflect back to inputs so user sees active range
    if (startInput) startInput.value = startDate;
    if (endInput) endInput.value = endDate;

    // Loading state
    thead.innerHTML = '<tr><th>Personel</th></tr>';
    tbody.innerHTML = '<tr><td class="text-center">Yükleniyor...</td></tr>';

    try {
        const params = new URLSearchParams();
        params.set('start_date', startDate);
        params.set('end_date', endDate);

        // Fetch all required data for the range
        const [persResp, perfResp, recResp, ahResp, aoResp, tfResp, wcResp, wcSumResp] = await Promise.all([
            fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/performance-records?${params.toString()}`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/daily-records?${params.toString()}`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/after-hours?${params.toString()}`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/attendance?${params.toString()}`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/training-feedback?${params.toString()}`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/warnings-cuts?${params.toString()}`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/warnings-cuts/summary?${params.toString()}`, { cache: 'no-store' })
        ]);

        const [persJson, perfJson, recJson, ahJson, aoJson, tfJson, wcJson, wcSumJson] = await Promise.all([
            persResp.ok ? persResp.json() : Promise.resolve({ data: [] }),
            perfResp.ok ? perfResp.json() : Promise.resolve({ data: [] }),
            recResp.ok ? recResp.json() : Promise.resolve({ data: [] }),
            ahResp.ok ? ahResp.json() : Promise.resolve({ data: [] }),
            aoResp.ok ? aoResp.json() : Promise.resolve({ data: [] }),
            tfResp.ok ? tfResp.json() : Promise.resolve({ data: [] }),
            wcResp.ok ? wcResp.json() : Promise.resolve({ data: [] }),
            wcSumResp.ok ? wcSumResp.json() : Promise.resolve({ data: {} })
        ]);

        const personnel = Array.isArray(persJson?.data) ? persJson.data : [];
        const perf = Array.isArray(perfJson?.data) ? perfJson.data : [];
        const records = Array.isArray(recJson?.data) ? recJson.data : [];
        const afterHours = Array.isArray(ahJson?.data) ? ahJson.data : [];
    const overrides = Array.isArray(aoJson?.data) ? aoJson.data : [];
    const trainingFeedback = Array.isArray(tfJson?.data) ? tfJson.data : [];
        const warningsCuts = Array.isArray(wcJson?.data) ? wcJson.data : [];
        const warningsCutsSummary = wcSumJson?.data || { per_person: [], type_totals: {}, daily_trend: {} };

        // Sort personnel: As Ekip first, then Paf, then name
        const sortedPersonnel = personnel.slice().sort((a, b) => {
            const ta = a.team === 'As Ekip' ? 0 : (a.team === 'Paf Ekip' ? 1 : 2);
            const tb = b.team === 'As Ekip' ? 0 : (b.team === 'Paf Ekip' ? 1 : 2);
            if (ta !== tb) return ta - tb;
            return (a.name || '').localeCompare(b.name || '', 'tr', { sensitivity: 'base' });
        });

        // Build date range array
        const days = buildDateRange(startDate, endDate);

        // Index presence and overrides
        const daytimeSet = new Set(); // key: pid|date
        const afterSet = new Set(); // key: pid|date
        const ovMap = new Map(); // key: pid|date -> override
        const key = (pid, d) => `${pid}|${d}`;

        for (const r of perf) if (r.personnel_id && r.date) daytimeSet.add(key(r.personnel_id, r.date));
        for (const r of records) if (r.personnel_id && r.date) daytimeSet.add(key(r.personnel_id, r.date));
        for (const r of afterHours) if (r.personnel_id && r.date) afterSet.add(key(r.personnel_id, r.date));
        for (const r of overrides) if (r.personnel_id && r.date) ovMap.set(key(r.personnel_id, r.date), r);
        try { window.__attendanceOverridesMap = ovMap; } catch {}

        // Render header
        const headerCells = [
            '<th class="sticky-col">Personel</th>',
            ...days.map(dt => `<th title="${dt}">${dt.split('-')[2]}</th>`),
            '<th>Toplam</th>',
            '<th>İzin</th>'
        ];
        thead.innerHTML = `<tr>${headerCells.join('')}</tr>`;

        // Render body rows with numeric values and izin totals (values < 1)
        const rowsHtml = sortedPersonnel.map(p => {
            const pid = p.id;
            let total = 0;
            let totalLeave = 0; // sum of values < 1 (0 => 1 day, 0.5 => 0.5 day)
            const cells = days.map(dt => {
                const hasAfter = afterSet.has(key(pid, dt));
                const ov = ovMap.get(key(pid, dt));
                const isFuture = dt > todayStr;
                let numeric = null;
                let isWeekly = false;
                if (ov && (ov.value === 0 || ov.value === 0.5 || ov.value === 1 || ov.value === 0.0)) {
                    numeric = Number(ov.value);
                    if ((ov.leave_type && String(ov.leave_type).toLowerCase() === 'weekly') || (ov.period && String(ov.period).toLowerCase() === 'weekly')) {
                        isWeekly = true;
                    }
                } else if (!isFuture) {
                    numeric = daytimeSet.has(key(pid, dt)) ? 1 : 0;
                } // else leave as null for future unknown

                if (numeric !== null) total += numeric;
                if (!isFuture) {
                    if (numeric === 0) totalLeave += 1;
                    else if (numeric === 0.5) totalLeave += 0.5;
                }

                const classes = [
                    numeric === null ? 'att-unknown' : numeric === 1 ? 'att-1' : numeric === 0.5 ? 'att-05' : 'att-0',
                    hasAfter ? 'mesai' : '',
                    isWeekly ? 'weekly-leave' : ''
                ].filter(Boolean).join(' ');
                const ovAttr = ov && ov.id != null ? ` data-override-id="${ov.id}"` : '';
                const text = numeric === null ? '-' : (numeric === 0.5 ? '0,5' : String(numeric));
                const titleBase = isWeekly ? `${text} (Haftalık izin)` : text;
                const title = hasAfter ? `${titleBase} (Mesai Sonrası)` : titleBase;
                return `<td class="${classes}" title="${title}"${ovAttr}>${text}</td>`;
            });
            const nameCell = `<td class="sticky-col"><span class="team-badge ${p.team === 'As Ekip' ? 'team-as' : (p.team === 'Paf Ekip' ? 'team-paf' : '')}" style="margin-right:6px;">${p.team || '-'}</span>${p.name || '-'}</td>`;
            const totalStr = Number.isInteger(total) ? String(total) : String(total).replace('.', ',');
            const leaveStr = Number.isInteger(totalLeave) ? String(totalLeave) : String(totalLeave).replace('.', ',');
            return `<tr data-personnel-id="${pid}">${nameCell}${cells.join('')}<td class="total-col">${totalStr}</td><td class="total-col">${leaveStr}</td></tr>`;
        }).join('');

        tbody.innerHTML = rowsHtml || '<tr><td class="text-center">Personel bulunamadı</td></tr>';

        // Render UYARI - KESİNTİ table under attendance
        try {
            const tfBody = document.getElementById('warningCutTableBody');
            const sumBody = document.getElementById('warningCutSummaryBody');
            if (sumBody) {
                const rows = (warningsCutsSummary.per_person || []).map(r => {
                    return `<tr>
                        <td>${r.personnel_name || '-'}</td>
                        <td>${r.uyari || 0}</td>
                        <td>${r.kesinti || 0}</td>
                        <td>${r.toplam || 0}</td>
                    </tr>`;
                }).join('');
                sumBody.innerHTML = rows || '<tr><td colspan="4" class="text-center">Kayıt yok</td></tr>';
            }
            if (tfBody) {
                // Prefer new endpoint data; fallback to trainingFeedback-backed entries
                let entries = Array.isArray(warningsCuts) && warningsCuts.length ? warningsCuts : (trainingFeedback || []).filter(r => r && (r.warning_interruption_type || r.warning_interruption_count));
                // Map for render, include derived fields and id
                const personById = {}; for (const p of personnel) personById[p.id] = p.name || `#${p.id}`;
                let mapped = entries.map(r => {
                    const typeField = (r.warning_interruption_type || r.type || '').toString().toLowerCase();
                    const countField = r.warning_interruption_count ?? r.count ?? r.feedback_count ?? 1;
                    return {
                        id: r.id,
                        date: r.date || '-',
                        personnel_id: r.personnel_id,
                        personnel_name: personById[r.personnel_id] || '-',
                        type: typeField === 'kesinti' ? 'Kesinti' : 'Uyarı',
                        raw_type: typeField || 'uyari',
                        count: Number(countField) || 1,
                        subject: r.warning_interruption_subject || r.subject || r.feedback_subject || '-'
                    };
                });

                // Sorting: small-to-large by last clicked header (persist in state)
                window.__wcSort = window.__wcSort || { key: 'date', dir: 'asc' };
                const sort = window.__wcSort;
                const collator = new Intl.Collator('tr', { numeric: true, sensitivity: 'base' });
                const dirMul = sort.dir === 'desc' ? -1 : 1;
                mapped.sort((a, b) => {
                    const ka = a[sort.key];
                    const kb = b[sort.key];
                    if (sort.key === 'count') return (Number(ka) - Number(kb)) * dirMul;
                    return collator.compare(String(ka), String(kb)) * dirMul;
                });

                if (!mapped.length) {
                    tfBody.innerHTML = '<tr><td colspan="6" class="text-center">Kayıt yok</td></tr>';
                } else {
                    const rows = mapped.map(r => {
                        return `<tr data-record-id="${r.id}">
                            <td>${r.date}</td>
                            <td>${r.personnel_name}</td>
                            <td>${r.type}</td>
                            <td>${r.count}</td>
                            <td>${r.subject}</td>
                            <td>
                                <button class="btn btn-sm btn-warning wc-edit-btn" data-record-id="${r.id}"><i class="fas fa-edit"></i></button>
                                <button class="btn btn-sm btn-danger wc-delete-btn" data-record-id="${r.id}"><i class="fas fa-trash"></i></button>
                            </td>
                        </tr>`;
                    }).join('');
                    tfBody.innerHTML = rows;
                }

                // Wire sorting click handlers once
                const table = document.getElementById('warningCutTable');
                if (table && !table.dataset.sortWired) {
                    table.dataset.sortWired = 'true';
                    table.querySelectorAll('th.wc-sort').forEach(th => {
                        th.addEventListener('click', () => {
                            const key = th.getAttribute('data-key');
                            if (!key) return;
                            if (window.__wcSort.key === key) {
                                window.__wcSort.dir = window.__wcSort.dir === 'asc' ? 'desc' : 'asc';
                            } else {
                                window.__wcSort.key = key; window.__wcSort.dir = 'asc';
                            }
                            updateAttendanceContent();
                        });
                    });
                }
                // Wire edit/delete buttons via delegation
                if (!table.dataset.actionsWired) {
                    table.dataset.actionsWired = 'true';
                    table.addEventListener('click', async (ev) => {
                        const editBtn = ev.target.closest && ev.target.closest('.wc-edit-btn');
                        const delBtn = ev.target.closest && ev.target.closest('.wc-delete-btn');
                        if (editBtn) {
                            const id = editBtn.getAttribute('data-record-id');
                            await loadPersonnelOptions('warning_cut_personnel_id');
                                const dataset = Array.isArray(warningsCuts) && warningsCuts.length ? warningsCuts : (trainingFeedback || []);
                                const rec = dataset.find(r => String(r.id) === String(id));
                            if (rec) {
                                const form = document.getElementById('warningCutForm');
                                if (form && typeof form.reset === 'function') form.reset();
                                document.getElementById('warning_cut_edit_id').value = String(rec.id);
                                document.getElementById('warning_cut_personnel_id').value = String(rec.personnel_id);
                                document.getElementById('warning_cut_date').value = rec.date || '';
                                    const t = (rec.warning_interruption_type || rec.type || 'uyari').toString().toLowerCase();
                                    const c = rec.warning_interruption_count ?? rec.count ?? 1;
                                    const s = rec.warning_interruption_subject || rec.subject || '';
                                    document.getElementById('warning_cut_type').value = t;
                                    document.getElementById('warning_cut_count').value = String(c);
                                    document.getElementById('warning_cut_subject').value = s;
                                openModal('warningCutModal');
                            }
                            ev.preventDefault();
                            return;
                        }
                        if (delBtn) {
                            const id = delBtn.getAttribute('data-record-id');
                            if (!id) return;
                            if (!confirm('Bu kaydı silmek istediğinize emin misiniz?')) return;
                            try {
                                // Prefer new endpoint for delete, fallback to training-feedback
                                let resp = await fetch(`${API_BASE_URL}/api/warnings-cuts/${id}`, { method: 'DELETE' });
                                if (!resp.ok) {
                                    resp = await fetch(`${API_BASE_URL}/api/training-feedback/${id}`, { method: 'DELETE' });
                                }
                                if (resp.ok) {
                                    showNotification('Kayıt silindi', 'success', 'İşlem Başarılı');
                                    updateAttendanceContent();
                                } else {
                                    const err = await resp.json().catch(()=>({}));
                                    showNotification(err.detail || 'Silme işlemi başarısız', 'error', 'Hata');
                                }
                            } catch (e) {
                                showNotification('Sunucuya ulaşılamadı', 'error', 'Hata');
                            }
                            ev.preventDefault();
                            return;
                        }
                    });
                }
            }
        } catch (tfErr) {
            console.warn('UYARI-KESİNTİ render skipped:', tfErr);
        }
    } catch (e) {
        console.error('❌ updateAttendanceContent failed:', e);
        tbody.innerHTML = '<tr><td class="text-center text-danger">Puantaj verileri yüklenemedi</td></tr>';
    }
}

async function exportWarningsCuts() {
    try {
        const startDate = document.getElementById('attendanceStartDate')?.value || '';
        const endDate = document.getElementById('attendanceEndDate')?.value || '';
        const params = new URLSearchParams();
        if (startDate) params.set('start_date', startDate);
        if (endDate) params.set('end_date', endDate);
    const url = `${API_BASE_URL}/api/warnings-cuts/export?${params.toString()}`;
    const resp = await fetch(url, { method: 'GET' });
        if (!resp.ok) {
            throw new Error(`Export failed: ${resp.status}`);
        }
        // Download file
        const blob = await resp.blob();
        const contentDisposition = resp.headers.get('Content-Disposition') || '';
        let fileName = 'warnings_cuts.xlsx';
        const match = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(contentDisposition);
        if (match) fileName = decodeURIComponent(match[1] || match[2] || fileName);
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        showNotification('Excel indirildi', 'success', 'İndirildi');
    } catch (e) {
        console.error('warnings-cuts export error', e);
        showNotification('Export sırasında hata oluştu', 'error', 'Hata');
    }
}

function updateDashboardContent() {
    // Update stat card labels per new requirements
    try {
        const setLabelForHeader = (headerId, labelText) => {
            const h = document.getElementById(headerId);
            if (h && h.parentElement) {
                const p = h.parentElement.querySelector('p');
                if (p) p.textContent = labelText;
            }
        };
    // 1) Aktif Personeller
    setLabelForHeader('totalPersonnel', 'Aktif Personeller');
    // 2) As/Paf ortalamaları (yalnızca değerler gösterilecek, açıklama gizlenir)
    setLabelForHeader('todayPerformance', '');
    // 3) Beklenen Değere Uzaklık
    setLabelForHeader('targetAchievement', 'Beklenen Değere Uzaklık');
    // 4) Kota Süreci
    setLabelForHeader('todayRecords', 'Kota Süreci');
    } catch (e) { /* noop */ }

    // Compute and render KPI values
    refreshDashboardStats();

    // Render dashboard summary table under KPIs (read-only, auto-updates from other pages)
    const dashboardCards = document.getElementById('dashboardCards');
    if (dashboardCards && !dashboardCards.dataset.initialized) {
        dashboardCards.innerHTML = `
            <div class="chart-card full-width">
                <h3>Dashboard</h3>
                <div class="page-actions" style="margin: 0 0 10px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                    <div class="date-range-filters">
                        <input type="date" id="dashboardStartDate" class="filter-input" placeholder="Başlangıç tarihi">
                        <span class="date-separator">-</span>
                        <input type="date" id="dashboardEndDate" class="filter-input" placeholder="Bitiş tarihi">
                        <button class="btn btn-primary btn-sm" id="applyDashboardFilter">
                            <i class="fas fa-filter"></i> Filtrele
                        </button>
                        <button class="btn btn-secondary btn-sm" id="clearDashboardFilter">
                            <i class="fas fa-times"></i> Temizle
                        </button>
                    </div>
                    <button class="btn btn-info btn-sm" id="exportDashboardExcelBtn">
                        <i class="fas fa-file-excel"></i> Excel İndir
                    </button>
                    <span id="dashboardFilterRange" class="filter-range-label" style="margin-left:auto;color:#666;font-size:12px;"></span>
                </div>
                <div class="table-container" style="background: rgba(255,255,255,0.95);">
                    <table class="data-table" id="dashboardSummaryTable">
                        <thead>
                            <tr>
                                <th class="sortable" data-key="team">Ekipler <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="name">Personeller <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="member_count">Üye Adedi <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="whatsapp_count">WhatsApp Adedi <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="device_count">Cihaz Adedi <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="unanswered_count">WhatsApp Cevapsız Adedi <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="knowledge_duel_total">Bilgi Düellosu <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="call_count">Çağrı Adedi <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                                <th class="sortable" data-key="call_score_avg">Çağrı Puanı <span class="sort-arrows"><span class="arrow-up">↑</span><span class="arrow-down">↓</span></span></th>
                            </tr>
                        </thead>
                        <tbody id="dashboardSummaryBody">
                            <tr><td colspan="9" class="text-center">Yükleniyor...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>`;
        dashboardCards.dataset.initialized = 'true';
        // Populate initial data
        try {
            // Wire filter & export handlers
            const applyBtn = document.getElementById('applyDashboardFilter');
            const clearBtn = document.getElementById('clearDashboardFilter');
            const exportBtn = document.getElementById('exportDashboardExcelBtn');
            const sdInput = document.getElementById('dashboardStartDate');
            const edInput = document.getElementById('dashboardEndDate');

            if (sdInput && edInput) {
                // restore from state if present
                if (dashboardSummaryState.startDate) sdInput.value = dashboardSummaryState.startDate;
                if (dashboardSummaryState.endDate) edInput.value = dashboardSummaryState.endDate;
            }

            if (applyBtn) {
                applyBtn.addEventListener('click', () => {
                    const sd = sdInput?.value || '';
                    const ed = edInput?.value || '';
                    if (sd && ed && sd > ed) {
                        showNotification('Başlangıç tarihi bitiş tarihinden büyük olamaz', 'warning', 'Geçersiz Tarih');
                        return;
                    }
                    dashboardSummaryState.startDate = sd;
                    dashboardSummaryState.endDate = ed;
                    updateDashboardSummaryTable();
                });
            }
            if (clearBtn) {
                clearBtn.addEventListener('click', () => {
                    if (sdInput) sdInput.value = '';
                    if (edInput) edInput.value = '';
                    dashboardSummaryState.startDate = '';
                    dashboardSummaryState.endDate = '';
                    updateDashboardSummaryTable();
                });
            }
            if (exportBtn) {
                exportBtn.addEventListener('click', async () => {
                    const sd = (document.getElementById('dashboardStartDate')?.value) || '';
                    const ed = (document.getElementById('dashboardEndDate')?.value) || '';
                    const params = new URLSearchParams();
                    if (sd) params.append('start_date', sd);
                    if (ed) params.append('end_date', ed);
                    const url = `${API_BASE_URL}/api/export/dashboard-summary-excel${params.toString() ? ('?' + params.toString()) : ''}`;
                    try {
                        const resp = await fetch(url, { method: 'GET' });
                        if (!resp.ok) throw new Error('HTTP ' + resp.status);
                        const blob = await resp.blob();
                        const dlUrl = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        const ts = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
                        let fname = 'dashboard_ozet';
                        if (sd || ed) fname += '_' + [sd, ed].filter(Boolean).join('_to_');
                        fname += `_${ts}.xlsx`;
                        a.href = dlUrl;
                        a.download = fname;
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        URL.revokeObjectURL(dlUrl);
                        showNotification('Excel indiriliyor', 'info', 'Rapor');
                    } catch (e) {
                        console.error('Excel export failed', e);
                        showNotification('Excel indirilemedi', 'error', 'Hata');
                    }
                });
            }
            wireDashboardSummarySorting();
            updateDashboardSummaryTable();
        } catch (e) { console.error('Dashboard summary init failed', e); }
    } else if (dashboardCards) {
        // Update data when dashboard is revisited
        try { updateDashboardSummaryTable(); } catch (e) { console.error('Dashboard summary refresh failed', e); }
    }
}

async function refreshDashboardStats() {
    try {
        const [persResp, perfResp, targetsResp] = await Promise.all([
            fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/performance-records`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/targets`, { cache: 'no-store' })
        ]);

        const [persJson, perfJson, targetsJson] = await Promise.all([
            persResp.ok ? persResp.json() : Promise.resolve({ data: [] }),
            perfResp.ok ? perfResp.json() : Promise.resolve({ data: [] }),
            targetsResp.ok ? targetsResp.json() : Promise.resolve({ data: [] })
        ]);

        const personnel = Array.isArray(persJson?.data) ? persJson.data : [];
        const performance = Array.isArray(perfJson?.data) ? perfJson.data : [];
        const targets = Array.isArray(targetsJson?.data) ? targetsJson.data : [];

        // 1) Aktif Personel (toplam personel sayısı)
        const totalEl = document.getElementById('totalPersonnel');
        if (totalEl) totalEl.textContent = String(personnel.length);

        // Prepare helpers
        const teamById = {};
        for (const p of personnel) teamById[p.id] = p.team || '';

        // 2) As/Paf ortalamaları (Ay başından bugüne - MTD)
        const now = new Date();
        const ym = now.toISOString().slice(0, 7); // YYYY-MM
        let mtdRecords = performance.filter(r => (r.date || '').startsWith(ym));
        if (mtdRecords.length === 0) {
            // Fallback: tüm performans kayıtları
            mtdRecords = performance;
        }
        // Sum team totals for the month, then divide by the last day with data (gün oranı)
        const totals = { 'As Ekip': 0, 'Paf Ekip': 0 };
        let lastDayWithData = 0;
        for (const r of mtdRecords) {
            const team = teamById[r.personnel_id] || '';
            const mc = parseFloat(r.member_count || 0) || 0;
            if (team === 'As Ekip') totals['As Ekip'] += mc;
            else if (team === 'Paf Ekip') totals['Paf Ekip'] += mc;
            // Track the latest day number in this month from available data
            if (r.date && r.date.startsWith(ym)) {
                const d = parseInt(r.date.split('-')[2], 10);
                if (!isNaN(d) && d > lastDayWithData) lastDayWithData = d;
            }
        }
        if (!lastDayWithData) lastDayWithData = now.getDate();
        const avgAs = lastDayWithData ? (totals['As Ekip'] / lastDayWithData) : 0;
        const avgPaf = lastDayWithData ? (totals['Paf Ekip'] / lastDayWithData) : 0;
        const perfEl = document.getElementById('todayPerformance');
        if (perfEl) {
            const fmt = (v) => {
                if (!Number.isFinite(v) || v <= 0) return '-';
                const rounded = Math.round(v * 10) / 10; // 1 decimal
                return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
            };
            perfEl.innerHTML = `
                <div class="kpi-dual">
                    <div class="kpi-row"><span class="kpi-label">As Ekip Ortalama:</span> <span class="kpi-value">${fmt(avgAs)}</span></div>
                    <div class="kpi-row"><span class="kpi-label">Paf Ekip Ortalama:</span> <span class="kpi-value">${fmt(avgPaf)}</span></div>
                </div>
            `;
            perfEl.style.display = 'block';
        }

    // 3) Beklenen Değere Uzaklık
    const asCount = personnel.filter(p => (p.team || '') === 'As Ekip').length;
    const pafCount = personnel.filter(p => (p.team || '') === 'Paf Ekip').length;
    const dailyExpected = (asCount * 7) + (pafCount * 5);
    const dayOfMonth = now.getDate();
    const expectedToDate = dailyExpected * dayOfMonth;
    const actualToDate = mtdRecords.reduce((sum, r) => sum + (parseFloat(r.member_count || 0) || 0), 0);
    const distance = Math.max(expectedToDate - actualToDate, 0); // uzaklık negatif olmaz varsayımı
    const targetAchEl = document.getElementById('targetAchievement');
    if (targetAchEl) targetAchEl.textContent = Number.isFinite(distance) ? String(Math.round(distance)) : '-';

        // 4) Kota Süreci -> Hedefler başlığında kaç personel varsa o adet
        const todayRecEl = document.getElementById('todayRecords');
        if (todayRecEl) {
            const uniqueTargetPersonnel = Array.from(new Set((targets || []).map(t => t.personnel_id))).length;
            todayRecEl.textContent = String(uniqueTargetPersonnel);
        }

        dashboardState.lastComputedAt = Date.now();
    } catch (e) {
        console.warn('⚠️ refreshDashboardStats failed:', e);
    }
}

// Build dashboard summary data from APIs and render rows according to sort
async function updateDashboardSummaryTable() {
    const tbody = document.getElementById('dashboardSummaryBody');
    if (!tbody) return;
    try {
        // Read filters
        const sdInput = document.getElementById('dashboardStartDate');
        const edInput = document.getElementById('dashboardEndDate');
        const sd = (sdInput?.value || dashboardSummaryState.startDate || '').trim();
        const ed = (edInput?.value || dashboardSummaryState.endDate || '').trim();
        if (sd && ed && sd > ed) {
            showNotification('Başlangıç tarihi bitiş tarihinden büyük olamaz', 'warning', 'Geçersiz Tarih');
            return;
        }
        dashboardSummaryState.startDate = sd;
        dashboardSummaryState.endDate = ed;

        // Update range label
        const rangeEl = document.getElementById('dashboardFilterRange');
        if (rangeEl) {
            rangeEl.textContent = (sd || ed) ? `Seçili aralık: ${sd || '…'} - ${ed || '…'}` : 'Seçili aralık: Tümü';
        }

        // Fetch all needed data in parallel
        const qp = new URLSearchParams();
        if (sd) qp.append('start_date', sd);
        if (ed) qp.append('end_date', ed);
        const q = qp.toString();
        const [persResp, perfResp, recResp] = await Promise.all([
            fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/performance-records${q ? ('?' + q) : ''}`, { cache: 'no-store' }),
            fetch(`${API_BASE_URL}/api/daily-records${q ? ('?' + q) : ''}`, { cache: 'no-store' })
        ]);

        const [persJson, perfJson, recJson] = await Promise.all([
            persResp.ok ? persResp.json() : Promise.resolve({ data: [] }),
            perfResp.ok ? perfResp.json() : Promise.resolve({ data: [] }),
            recResp.ok ? recResp.json() : Promise.resolve({ data: [] })
        ]);

        const personnel = Array.isArray(persJson?.data) ? persJson.data : [];
        const perf = Array.isArray(perfJson?.data) ? perfJson.data : [];
        const records = Array.isArray(recJson?.data) ? recJson.data : [];

        // Aggregate performance per personnel
        const perfAgg = {};
        for (const r of perf) {
            const pid = r.personnel_id;
            if (!perfAgg[pid]) {
                perfAgg[pid] = {
                    member_count: 0,
                    whatsapp_count: 0,
                    device_count: 0,
                    unanswered_count: 0,
                    knowledge_duel_sum: 0,
                    perf_count: 0
                };
            }
            perfAgg[pid].member_count += Number(r.member_count || 0);
            perfAgg[pid].whatsapp_count += Number(r.whatsapp_count || 0);
            perfAgg[pid].device_count += Number(r.device_count || 0);
            perfAgg[pid].unanswered_count += Number(r.unanswered_count || 0);
            perfAgg[pid].knowledge_duel_sum += Number(r.knowledge_duel_result || 0);
            perfAgg[pid].perf_count += 1;
        }

        // Aggregate call records per personnel
        const recAgg = {};
        for (const r of records) {
            const pid = r.personnel_id;
            if (!recAgg[pid]) {
                recAgg[pid] = { call_count: 0, score_sum: 0 };
            }
            recAgg[pid].call_count += 1;
            recAgg[pid].score_sum += Number(r.score || 0);
        }

        // Build raw row objects
    const rows = personnel.map(p => {
            const pid = p.id;
            const pa = perfAgg[pid] || { member_count: 0, whatsapp_count: 0, device_count: 0, unanswered_count: 0, knowledge_duel_sum: 0, perf_count: 0 };
            const ra = recAgg[pid] || { call_count: 0, score_sum: 0 };
            const kdTotal = Number(pa.knowledge_duel_sum || 0);
            const avgScore = ra.call_count ? (ra.score_sum / ra.call_count) : null;
            return {
                team: p.team || '',
                name: p.name || '-',
                member_count: pa.member_count || 0,
                whatsapp_count: pa.whatsapp_count || 0,
                device_count: pa.device_count || 0,
                unanswered_count: pa.unanswered_count || 0,
                knowledge_duel_total: kdTotal,
                call_count: ra.call_count || 0,
                call_score_avg: avgScore
            };
        });

        // Save and render with current sort
        dashboardSummaryState.rows = rows;
        renderDashboardSummaryRows();
    } catch (err) {
        console.error('❌ updateDashboardSummaryTable failed:', err);
        tbody.innerHTML = `<tr><td colspan="9" class="text-center text-danger">Özet veriler yüklenemedi</td></tr>`;
    }
}

// Sort and render rows into tbody according to dashboardSummaryState
function renderDashboardSummaryRows() {
    const tbody = document.getElementById('dashboardSummaryBody');
    if (!tbody) return;
    const rows = (dashboardSummaryState.rows || []).slice();

    const teamOrder = { 'As Ekip': 0, 'Paf Ekip': 1 };

    const { sortKey, sortDir } = dashboardSummaryState;
    const dir = sortDir === 'desc' ? -1 : 1;

    rows.sort((a, b) => {
        if (sortKey === 'default') {
            const ta = teamOrder[a.team] ?? 2;
            const tb = teamOrder[b.team] ?? 2;
            if (ta !== tb) return (ta - tb) * dir; // allow dir toggle on default too
            return (a.name || '').localeCompare(b.name || '', 'tr') * dir;
        }
        if (['team','name'].includes(sortKey)) {
            if (sortKey === 'team') {
                const ta = teamOrder[a.team] ?? 2;
                const tb = teamOrder[b.team] ?? 2;
                if (ta !== tb) return (ta - tb) * dir;
                // fallback to name to stabilize
                return (a.name || '').localeCompare(b.name || '', 'tr') * dir;
            }
            return (a[sortKey] || '').localeCompare(b[sortKey] || '', 'tr') * dir;
        }
        // numeric keys
        const av = a[sortKey];
        const bv = b[sortKey];
        const an = (av === null || av === undefined || av === '-') ? NaN : Number(av);
        const bn = (bv === null || bv === undefined || bv === '-') ? NaN : Number(bv);
        // put NaN at the end for asc, start for desc
        const aIsNaN = Number.isNaN(an);
        const bIsNaN = Number.isNaN(bn);
        if (aIsNaN && bIsNaN) return 0;
        if (aIsNaN) return sortDir === 'asc' ? 1 : -1;
        if (bIsNaN) return sortDir === 'asc' ? -1 : 1;
        if (an === bn) return 0;
        return an > bn ? dir : -dir;
    });

    // Build HTML
    const html = rows.map(r => {
        const teamClass = r.team === 'As Ekip' ? 'dot-as' : (r.team === 'Paf Ekip' ? 'dot-paf' : '');
    const kd = (r.knowledge_duel_total === null || Number.isNaN(Number(r.knowledge_duel_total))) ? '-' : String(Number(r.knowledge_duel_total));
        const cs = (r.call_score_avg === null || Number.isNaN(Number(r.call_score_avg))) ? '-' : (Math.round(r.call_score_avg * 100) / 100).toFixed(2);
        return `
            <tr>
                <td>${r.team ? `<span class="team-chip team-glass"><span class="dot ${teamClass}"></span>${r.team}</span>` : '-'}</td>
                <td>${r.name}</td>
                <td>${r.member_count}</td>
                <td>${r.whatsapp_count}</td>
                <td>${r.device_count}</td>
                <td>${r.unanswered_count}</td>
                <td>${kd}</td>
                <td>${r.call_count}</td>
                <td>${cs}</td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = html || `<tr><td colspan="9" class="text-center">Personel bulunamadı</td></tr>`;
    updateDashboardSortUI();
}

// (duplicate Puantaj functions removed)

// Attach click handlers to sortable headers
function wireDashboardSummarySorting() {
    const table = document.getElementById('dashboardSummaryTable');
    if (!table) return;
    const headers = table.querySelectorAll('th.sortable');
    headers.forEach(th => {
        th.addEventListener('click', () => {
            const key = th.getAttribute('data-key');
            if (!key) return;
            if (dashboardSummaryState.sortKey === key || (dashboardSummaryState.sortKey === 'default' && key === 'team')) {
                // toggle direction
                dashboardSummaryState.sortDir = dashboardSummaryState.sortDir === 'asc' ? 'desc' : 'asc';
                // if we were on default and clicked team, set sortKey to team to lock
                if (dashboardSummaryState.sortKey === 'default') dashboardSummaryState.sortKey = key;
            } else {
                dashboardSummaryState.sortKey = key;
                dashboardSummaryState.sortDir = 'asc';
            }
            renderDashboardSummaryRows();
        });
    });
    // initialize UI state on first wire
    updateDashboardSortUI();
}

// Reflect current sort state in header arrow styles
function updateDashboardSortUI() {
    const table = document.getElementById('dashboardSummaryTable');
    if (!table) return;
    const headers = table.querySelectorAll('th.sortable');
    headers.forEach(th => {
        th.classList.remove('active','asc','desc');
        const key = th.getAttribute('data-key');
        // default state highlights team header
        const isActive = (dashboardSummaryState.sortKey === key) || (dashboardSummaryState.sortKey === 'default' && key === 'team');
        if (isActive) {
            th.classList.add('active');
            th.classList.add(dashboardSummaryState.sortDir === 'desc' ? 'desc' : 'asc');
        }
    });
}

async function updatePersonnelContent() {
    console.log('🔄 updatePersonnelContent() called');
    const tableBody = document.querySelector('#personnelTable tbody');
    console.log('📋 Table body element found:', !!tableBody);
    
    if (tableBody) {
        try {
            console.log('🔄 Fetching personnel data from API (no-cache)...');
            const personnelUrl = `${API_BASE_URL}/api/personnel?ts=${Date.now()}`;
            const response = await fetch(personnelUrl, {
                cache: 'no-store',
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            console.log('📡 API Response status:', response.status);
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Personnel data loaded:', result.data);
                console.log('📊 Personnel count:', result.data ? result.data.length : 0);
                
                if (result.data && result.data.length > 0) {
                    // EKIP BAZINDA SIRALAMA: As Ekip üstte, Paf Ekip altta
                    const sortedPersonnel = result.data.sort((a, b) => {
                        // As Ekip önce, Paf Ekip sonra
                        if (a.team === 'As Ekip' && b.team === 'Paf Ekip') return -1;
                        if (a.team === 'Paf Ekip' && b.team === 'As Ekip') return 1;
                        // Aynı ekipte ise isme göre sırala
                        return a.name.localeCompare(b.name, 'tr', { sensitivity: 'base' });
                    });
                    
                    tableBody.innerHTML = sortedPersonnel.map(person => `
                        <tr data-personnel-id="${person.id}">
                            <td>${person.name || '-'}</td>
                            <td>${person.username || '-'}</td>
                            <td>${person.email || '-'}</td>
                            <td><span class="team-badge ${person.team === 'As Ekip' ? 'team-as' : 'team-paf'}">${person.team || '-'}</span></td>
                            <td>${person.hire_date || '-'}</td>
                            <td>${person.reference || '-'}</td>
                            <td>${person.promotion_date || '-'}</td>
                            <td>
                                <button class="btn-warning btn-sm edit-btn" data-record-id="${person.id}" title="Personeli Düzenle" style="margin-right: 5px;">
                                    <i class="fas fa-edit"></i>
                                </button>
                                <button class="btn-danger btn-sm delete-btn" data-record-id="${person.id}" title="Personeli Sil">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </td>
                        </tr>
                    `).join('');
                    
                    console.log('✅ Table updated successfully with new HTML content');
                } else {
                    tableBody.innerHTML = `
                        <tr>
                            <td colspan="8" class="text-center">Henüz personel kaydı yok</td>
                        </tr>
                    `;
                }
            } else {
                throw new Error('API response not OK');
            }
        } catch (error) {
            console.error('❌ Error loading personnel:', error);
            tableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-danger">Personel verileri yüklenemedi</td>
                </tr>
            `;
        }
    }
}

// Records sayfası: personel dropdownını doldur ve tarih/personel filtresini uygula
async function populateRecordsPersonnelFilter() {
    const personnelFilter = document.getElementById('personnelFilter');
    if (!personnelFilter) return;
    try {
        const response = await fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' });
        if (!response.ok) return;
        const result = await response.json();
        const currentValue = personnelFilter.value;
        personnelFilter.innerHTML = '<option value="">Tüm Personel</option>';
        (result.data || []).forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            personnelFilter.appendChild(opt);
        });
        personnelFilter.value = currentValue;
    } catch (e) {
        console.error('❌ Failed to populate records personnel filter:', e);
    }
}

async function applyRecordsDateFilter() {
    console.log('📅 Applying records date/personnel filter...');
    const startDate = document.getElementById('recordsStartDate')?.value || '';
    const endDate = document.getElementById('recordsEndDate')?.value || '';
    const personnelId = document.getElementById('personnelFilter')?.value || '';
    if (startDate && endDate && startDate > endDate) {
        alert('⚠️ Başlangıç tarihi bitiş tarihinden büyük olamaz!');
        return;
    }
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (personnelId) params.append('personnel_id', personnelId);
    try {
        const url = `${API_BASE_URL}/api/daily-records?${params.toString()}`;
        const resp = await fetch(url, { cache: 'no-store' });
        if (!resp.ok) throw new Error('API not OK');
        const result = await resp.json();
        // Reuse existing renderer with injected data
        await renderRecordsWithData(result.data);
    } catch (e) {
        console.error('❌ Filter error:', e);
        showNotification('❌ Kayıtlar filtrelenemedi', 'error');
    }
}

function clearRecordsDateFilterUI() {
    const ids = ['recordsStartDate','recordsEndDate','personnelFilter'];
    ids.forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
}

async function clearRecordsDateFilter() {
    clearRecordsDateFilterUI();
    await updateRecordsContent();
}

async function updatePerformanceContent() {
    console.log('🏆 updatePerformanceContent() called');

    const detailTableBody = document.querySelector('#performanceTableBody');
    const summaryTableBody = document.querySelector('#performanceSummaryTableBody');
    console.log('📋 Performance detail tbody exists:', !!detailTableBody, 'summary tbody exists:', !!summaryTableBody);

    try {
        console.log('🔄 Fetching performance records from API (no-cache)...');
        const response = await fetch(`${API_BASE_URL}/api/performance-records?ts=${Date.now()}`, {
            cache: 'no-store',
            headers: { 'Cache-Control': 'no-cache' }
        });
        console.log('📡 Performance API Response status:', response.status);

        if (!response.ok) throw new Error(`API response not OK (${response.status})`);

        const result = await response.json();
        const records = Array.isArray(result?.data) ? result.data : [];
        console.log('✅ Performance records loaded:', records.length);

        // Update detail table
        if (detailTableBody) {
            if (records.length > 0) {
        detailTableBody.innerHTML = records.map(record => `
                    <tr data-record-id="${record.id}">
                        <td>${record.date}</td>
                        <td>${record.personnel_name || 'Bilinmeyen Personel'}</td>
                        <td>${record.member_count || 0}</td>
                        <td>${record.whatsapp_count || 0}</td>
                        <td>${record.device_count || 0}</td>
                        <td>${record.unanswered_count || 0}</td>
                        <td>${record.knowledge_duel_result || 0}</td>
            <td>${(record.reward_penalty || '').toString().toLowerCase() === 'odul' ? 'ödül' : '-'}</td>
            <td>${(record.reward_penalty || '').toString().toLowerCase() === 'ceza' ? 'ceza' : '-'}</td>
                        <td>
                            <button class="btn-warning btn-sm edit-btn" data-record-id="${record.id}" title="Kaydı Düzenle" style="margin-right: 5px;">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="btn-danger btn-sm delete-btn" data-record-id="${record.id}" title="Kaydı Sil">
                                <i class="fas fa-trash"></i>
                            </button>
                        </td>
                    </tr>
                `).join('');
            } else {
                detailTableBody.innerHTML = `
                    <tr>
                        <td colspan="10" class="text-center">Henüz performans kaydı yok</td>
                    </tr>
                `;
            }
        }

        // Update summary table
        if (summaryTableBody) {
            if (records.length > 0) {
                const summaryByPersonnel = {};
                for (const record of records) {
                    const pid = record.personnel_id;
                    if (!summaryByPersonnel[pid]) {
                        summaryByPersonnel[pid] = {
                            personnel_id: pid,
                            personnel_name: record.personnel_name || `Personel ${pid}`,
                            member_count: 0,
                            whatsapp_count: 0,
                            device_count: 0,
                            unanswered_count: 0,
                knowledge_duel_result: 0,
                reward_count: 0,
                penalty_count: 0,
                record_count: 0
                        };
                    }
                    summaryByPersonnel[pid].member_count += Number(record.member_count || 0);
                    summaryByPersonnel[pid].whatsapp_count += Number(record.whatsapp_count || 0);
                    summaryByPersonnel[pid].device_count += Number(record.device_count || 0);
                    summaryByPersonnel[pid].unanswered_count += Number(record.unanswered_count || 0);
                    summaryByPersonnel[pid].knowledge_duel_result += Number(record.knowledge_duel_result || 0);
            const rp = (record.reward_penalty || '').toString().toLowerCase();
            if (rp === 'odul') summaryByPersonnel[pid].reward_count += 1;
            else if (rp === 'ceza') summaryByPersonnel[pid].penalty_count += 1;
                    summaryByPersonnel[pid].record_count += 1;
                }

                summaryTableBody.innerHTML = Object.values(summaryByPersonnel).map(summary => {
                    const kdTotal = Number(summary.knowledge_duel_result || 0);
                    return `
                        <tr>
                            <td>${summary.personnel_name}</td>
                            <td>${summary.member_count}</td>
                            <td>${summary.whatsapp_count}</td>
                            <td>${summary.device_count}</td>
                            <td>${summary.unanswered_count}</td>
                            <td>${kdTotal}</td>
                            <td>${summary.reward_count || 0}</td>
                            <td>${summary.penalty_count || 0}</td>
                        </tr>
                    `;
                }).join('');
            } else {
                summaryTableBody.innerHTML = `
                    <tr>
                        <td colspan="8" class="text-center">Henüz performans özeti yok</td>
                    </tr>
                `;
            }
        }
    } catch (error) {
        console.error('❌ Error loading performance content:', error);
        if (detailTableBody) {
            detailTableBody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center text-danger">Performans kayıtları yüklenemedi</td>
                </tr>
            `;
        }
        const summaryTableBody = document.querySelector('#performanceSummaryTableBody');
        if (summaryTableBody) {
        summaryTableBody.innerHTML = `
                <tr>
            <td colspan="8" class="text-center text-danger">Performans özeti yüklenemedi</td>
                </tr>
            `;
        }
    }

    // Refresh filter dropdown
    await populatePerformancePersonnelFilter();
    // Update dashboard KPIs since performance affects them
    try { updateDashboardContent(); } catch {}
}

// Performans sayfası personel filtre dropdown'unu doldur
async function populatePerformancePersonnelFilter() {
    console.log('👥 Populating performance personnel filter...');
    
    const personnelFilter = document.getElementById('performancePersonnelFilter');
    if (personnelFilter) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/personnel`);
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Personnel data for filter loaded:', result.data);
                
                // Mevcut seçimi sakla
                const currentValue = personnelFilter.value;
                
                // Dropdown'u temizle ve varsayılan option'ı ekle
                personnelFilter.innerHTML = '<option value="">Tüm Personel</option>';
                
                // Personelleri ekle (team bazında sıralama ile)
                if (result.data && result.data.length > 0) {
                    // Team bazında sıralama: As Ekip önce, Paf Ekip sonra
                    const sortedPersonnel = [...result.data].sort((a, b) => {
                        const teamOrder = { 'As Ekip': 0, 'Paf Ekip': 1 };
                        const teamA = teamOrder[a.team] !== undefined ? teamOrder[a.team] : 2;
                        const teamB = teamOrder[b.team] !== undefined ? teamOrder[b.team] : 2;
                        
                        if (teamA !== teamB) {
                            return teamA - teamB;
                        }
                        return a.name.localeCompare(b.name, 'tr');
                    });
                    
                    sortedPersonnel.forEach(personnel => {
                        const option = document.createElement('option');
                        option.value = personnel.id;
                        option.textContent = `${personnel.name} (${personnel.team || 'Bilinmiyor'})`;
                        personnelFilter.appendChild(option);
                    });
                }
                
                // Seçimi geri yükle
                personnelFilter.value = currentValue;
                
            } else {
                console.error('❌ Failed to load personnel for filter');
            }
        } catch (error) {
            console.error('❌ Error populating personnel filter:', error);
        }
    }
}

async function updateTrainingFeedbackContent() {
    console.log('🎓 updateTrainingFeedbackContent() called');
    
    // Detaylı eğitim-geribildirim kayıtları tablosunu güncelle
    const detailTableBody = document.querySelector('#trainingFeedbackDetailTBody');
    console.log('📋 Training feedback table body element found:', !!detailTableBody);
    
    // Aktif tarih filtrelerini oku
    const tfStart = document.getElementById('trainingFeedbackStartDate')?.value || '';
    const tfEnd = document.getElementById('trainingFeedbackEndDate')?.value || '';
    const tfParams = new URLSearchParams();
    if (tfStart) tfParams.append('start_date', tfStart);
    if (tfEnd) tfParams.append('end_date', tfEnd);

    if (detailTableBody) {
        try {
            console.log('🔄 Fetching training-feedback records from API...');
            const response = await fetch(`${API_BASE_URL}/api/training-feedback${tfParams.toString() ? `?${tfParams.toString()}` : ''}`, { cache: 'no-store' });
            console.log('📡 Training feedback API Response status:', response.status);
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Training-feedback records loaded:', result.data);
                console.log('📊 Training feedback records count:', result.data ? result.data.length : 0);
                
                if (result.data && result.data.length > 0) {
                    console.log('📄 Updating training feedback table with', result.data.length, 'records');

                    detailTableBody.innerHTML = result.data.map((record) => {
                        const editBtnId = `edit-btn-${record.id}`;
                        const deleteBtnId = `delete-btn-${record.id}`;
                        return `
                            <tr data-record-id="${record.id}">
                                <td>${record.date || '-'}</td>
                                <td>${record.personnel_name || 'Bilinmeyen Personel'}</td>
                                <td>${record.general_training_count || 0}</td>
                                <td>${record.general_training_subject || '-'}</td>
                                <td>${record.personal_training_count || 0}</td>
                                <td>${record.personal_training_subject || '-'}</td>
                                <td>${record.feedback_count || 0}</td>
                                <td>${record.feedback_subject || '-'}</td>
                                <td>
                                    <button id="${editBtnId}" class="btn-warning btn-sm edit-btn" data-record-id="${record.id}" title="Kaydı Düzenle" style="margin-right: 5px; cursor: pointer; padding: 8px 12px;">
                                        <i class="fas fa-edit"></i> Düzenle
                                    </button>
                                    <button id="${deleteBtnId}" class="btn-danger btn-sm delete-btn" data-record-id="${record.id}" title="Kaydı Sil">
                                        <i class="fas fa-trash"></i> Sil
                                    </button>
                                </td>
                            </tr>
                        `;
                    }).join('');
                    
                    // Event listeners are handled centrally by a single global click handler.
                    // Avoid attaching per-button listeners here to prevent duplicate prompts.
                } else {
                    detailTableBody.innerHTML = `
                        <tr>
                            <td colspan="9" class="text-center">Henüz eğitim kaydı yok</td>
                        </tr>
                    `;
                }
            } else {
                throw new Error('API response not OK');
            }
        } catch (error) {
            console.error('❌ Error loading training-feedback records:', error);
            detailTableBody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center text-danger">Eğitim kayıtları yüklenemedi</td>
                </tr>
            `;
        }
    }
    
    // Özet tablosunu güncelle
    const summaryTableBody = document.querySelector('#trainingFeedbackSummaryTBody');
    if (summaryTableBody) {
        try {
            console.log('🔄 Fetching training-feedback summary...');
            const response = await fetch(`${API_BASE_URL}/api/training-feedback/summary${tfParams.toString() ? `?${tfParams.toString()}` : ''}`, { cache: 'no-store' });
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Training-feedback summary data:', result.data);
                
                if (result.data && result.data.length > 0) {
                    summaryTableBody.innerHTML = result.data.map(summary => {
                        return `
                            <tr>
                                <td>${summary.personnel_name || 'Bilinmeyen Personel'}</td>
                                <td class="clickable-cell" onclick="showTrainingFeedbackBreakdown(${summary.personnel_id}, 'general_training')" style="cursor: pointer; color: #007bff; text-decoration: underline;">${summary.general_training_count || 0}</td>
                                <td class="clickable-cell" onclick="showTrainingFeedbackBreakdown(${summary.personnel_id}, 'personal_training')" style="cursor: pointer; color: #007bff; text-decoration: underline;">${summary.personal_training_count || 0}</td>
                                <td class="clickable-cell" onclick="showTrainingFeedbackBreakdown(${summary.personnel_id}, 'feedback')" style="cursor: pointer; color: #007bff; text-decoration: underline;">${summary.feedback_count || 0}</td>
                            </tr>
                        `;
                    }).join('');
                } else {
                    summaryTableBody.innerHTML = `
                        <tr>
                            <td colspan="4" class="text-center">Henüz eğitim kaydı yok</td>
                        </tr>
                    `;
                }
            } else {
                throw new Error('API response not OK');
            }
        } catch (error) {
            console.error('❌ Error loading training-feedback summary:', error);
            summaryTableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center text-danger">Özet veriler yüklenemedi</td>
                </tr>
            `;
        }
    }

    // Personnel dropdown'ını doldur
    await populateTrainingFeedbackPersonnelDropdown();
    
    // Edit/Delete buttons are handled by the global event delegation; nothing to attach here.
    
    // Manuel test için - console'da çalıştırılabilir
    window.testEditButton = function() {
        console.log('🧪 Testing edit button manually...');
        const editBtns = document.querySelectorAll('.edit-btn');
        console.log('Found edit buttons:', editBtns.length);
        editBtns.forEach((btn, index) => {
            console.log(`Button ${index}:`, btn);
            console.log(`  - data-record-id: ${btn.getAttribute('data-record-id')}`);
            console.log(`  - classes: ${btn.className}`);
        });
        
        if (editBtns.length > 0) {
            console.log('🔥 Manually triggering first edit button...');
            const recordId = parseInt(editBtns[0].getAttribute('data-record-id'));
            editTrainingFeedbackRecord(recordId);
        }
    };
}

async function populateTrainingFeedbackPersonnelDropdown() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/personnel`);
        if (response.ok) {
            const result = await response.json();
            const dropdown = document.getElementById('training_feedback_personnel_id');
            if (dropdown) {
                dropdown.innerHTML = '<option value="">Personel Seçin</option>';
                result.data.forEach(person => {
                    dropdown.innerHTML += `<option value="${person.id}">${person.name}</option>`;
                });
            }
        }
    } catch (error) {
        console.error('❌ Error populating training-feedback personnel dropdown:', error);
    }
}

async function updateRecordsContent() {
    console.log('📞 updateRecordsContent() called');
    
    const tableBody = document.querySelector('#recordsTable tbody');
    console.log('📋 Records table body element found:', !!tableBody);
    
    if (tableBody) {
        try {
            console.log('🔄 Fetching daily records from API...');
            // Önce personel dropdown'ını doldur
            await populateRecordsPersonnelFilter();
            const response = await fetch(`${API_BASE_URL}/api/daily-records`, { cache: 'no-store' });
            console.log('📡 Records API Response status:', response.status);
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Daily records loaded:', result.data);
                console.log('📊 Daily records count:', result.data ? result.data.length : 0);
                
                if (result.data && result.data.length > 0) {
                    console.log('📋 Updating records table with', result.data.length, 'records');
                    // Personel listesini al
                    const personnelResponse = await fetch(`${API_BASE_URL}/api/personnel`);
                    let personnelMap = {};
                    if (personnelResponse.ok) {
                        const personnelResult = await personnelResponse.json();
                        personnelMap = personnelResult.data.reduce((map, person) => {
                            map[person.id] = person.name;
                            return map;
                        }, {});
                    }
                    
                    tableBody.innerHTML = result.data.map(record => `
                        <tr data-record-id="${record.id}">
                            <td>${record.date}</td>
                            <td>${personnelMap[record.personnel_id] || `Personel ${record.personnel_id}`}</td>
                            <td>${record.call_number}</td>
                            <td>${record.score}</td>
                            <td>${record.notes || '-'}</td>
                            <td>
                                <button class="btn-warning btn-sm edit-btn" data-record-id="${record.id}" title="Kaydı Düzenle" style="margin-right: 5px;">
                                    <i class="fas fa-edit"></i>
                                </button>
                                <button class="btn-danger btn-sm delete-btn" data-record-id="${record.id}" title="Kaydı Sil">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </td>
                        </tr>
                    `).join('');
                    
                    // Personel özet tablosunu güncelle
                    updatePersonnelSummaryTable(result.data, personnelMap);
                } else {
                    tableBody.innerHTML = `
                        <tr>
                            <td colspan="6" class="text-center">Henüz çağrı kaydı yok</td>
                        </tr>
                    `;
                    
                    // Özet tablosunu da temizle
                    updatePersonnelSummaryTable([], {});
                }
            } else {
                throw new Error('API response not OK');
            }
        } catch (error) {
            console.error('❌ Error loading daily records:', error);
            tableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center text-danger">Çağrı kayıtları yüklenemedi</td>
                </tr>
            `;
        }
    }
    
    // Çağrı puanı formunda personel seçeneklerini yükle
    console.log('🔄 Loading personnel options for call records form...');
    
    // Personel select elementini bul ve doldur
    const personnelSelects = [
        'callPersonnelId',
        'recordPersonnelId',
        'personnel_select',
        'personnel_id'
    ];
    
    personnelSelects.forEach(selectId => {
        const selectElement = document.getElementById(selectId);
        if (selectElement) {
            console.log(`📋 Found personnel select: ${selectId}`);
            loadPersonnelOptions(selectId);
        }
    });
    
    // Ayrıca tüm personnel select'leri bul (class ile)
    const personnelSelectsByClass = document.querySelectorAll('.personnel-select, select[name="personnel_id"], select[name*="personnel"]');
    personnelSelectsByClass.forEach((select, index) => {
        if (select.id) {
            console.log(`📋 Found personnel select by class/name: ${select.id}`);
            loadPersonnelOptions(select.id);
        } else {
            // ID yoksa geçici bir ID ver
            select.id = `personnel_select_${index}`;
            console.log(`📋 Created ID for personnel select: ${select.id}`);
            loadPersonnelOptions(select.id);
        }
    });
}

// Güncellediğimiz tablo render'ını tekrar kullanmak için küçük bir yardımcı
async function renderRecordsWithData(data) {
    const tableBody = document.querySelector('#recordsTable tbody');
    if (!tableBody) return;
    try {
        // Personel isimleri için harita
        const personnelResponse = await fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' });
        let personnelMap = {};
        if (personnelResponse.ok) {
            const pr = await personnelResponse.json();
            personnelMap = (pr.data || []).reduce((m, p) => { m[p.id] = p.name; return m; }, {});
        }
        if (data && data.length > 0) {
            tableBody.innerHTML = data.map(record => `
                <tr data-record-id="${record.id}">
                    <td>${record.date}</td>
                    <td>${personnelMap[record.personnel_id] || `Personel ${record.personnel_id}`}</td>
                    <td>${record.call_number}</td>
                    <td>${record.score}</td>
                    <td>${record.notes || '-'}</td>
                    <td>
                        <button class="btn-warning btn-sm edit-btn" data-record-id="${record.id}" title="Kaydı Düzenle" style="margin-right: 5px;">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn-danger btn-sm delete-btn" data-record-id="${record.id}" title="Kaydı Sil">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `).join('');
            updatePersonnelSummaryTable(data, personnelMap);
        } else {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center">Henüz çağrı kaydı yok</td>
                </tr>
            `;
            updatePersonnelSummaryTable([], {});
        }
    } catch (e) {
        console.error('❌ renderRecordsWithData failed:', e);
    }
}

// Personel özet tablosunu güncelle
function updatePersonnelSummaryTable(dailyRecords, personnelMap) {
    const summaryTableBody = document.getElementById('personnelSummaryTableBody');
    if (!summaryTableBody) return;
    
    console.log('📊 Updating personnel summary table...');
    
    if (!dailyRecords || dailyRecords.length === 0) {
        summaryTableBody.innerHTML = `
            <tr>
                <td colspan="3" class="text-center">Henüz çağrı özeti yok</td>
            </tr>
        `;
        return;
    }
    
    // Personel bazında özet hesapla
    const personnelSummary = {};
    
    dailyRecords.forEach(record => {
        const personnelId = record.personnel_id;
        const score = parseFloat(record.score) || 0;
        
        if (!personnelSummary[personnelId]) {
            personnelSummary[personnelId] = {
                name: personnelMap[personnelId] || `Personel ${personnelId}`,
                totalCalls: 0,
                totalScore: 0,
                averageScore: 0
            };
        }
        
        personnelSummary[personnelId].totalCalls++;
        personnelSummary[personnelId].totalScore += score;
        personnelSummary[personnelId].averageScore = personnelSummary[personnelId].totalScore / personnelSummary[personnelId].totalCalls;
    });
    
    // Tabloyu doldur
    summaryTableBody.innerHTML = Object.values(personnelSummary).map(summary => `
        <tr>
            <td>${summary.name}</td>
            <td>${summary.totalCalls}</td>
            <td>${summary.averageScore.toFixed(2)}</td>
        </tr>
    `).join('');
    
    console.log('✅ Personnel summary table updated:', personnelSummary);
}

async function updateAfterHoursContent() {
    console.log('🕒 updateAfterHoursContent() called');
    const sumBody = document.getElementById('afterHoursSummaryBody');
    const tableBody = document.getElementById('afterHoursTableBody');
    const startDate = document.getElementById('afterHoursStartDate')?.value || '';
    const endDate = document.getElementById('afterHoursEndDate')?.value || '';
    try {
        // Summary
        const sParams = new URLSearchParams();
        if (startDate) sParams.append('start_date', startDate);
        if (endDate) sParams.append('end_date', endDate);
        if (sumBody) sumBody.innerHTML = `<tr><td colspan="5" class="text-center">Yükleniyor...</td></tr>`;
        const sumResp = await fetch(`${API_BASE_URL}/api/after-hours/summary${sParams.toString() ? `?${sParams.toString()}` : ''}`, { cache: 'no-store' });
        if (sumResp.ok) {
            const js = await sumResp.json();
            const d = js?.data || { total_call_count: 0, total_talk_duration: 0, total_member_count: 0, record_count: 0 };
            // Attempt to determine personnel name from current visible detail list
            let personnelNameForSummary = '';
            try {
                const cachedFirstRow = document.querySelector('#afterHoursTableBody tr[data-record-id]');
                if (cachedFirstRow) {
                    const nameCell = cachedFirstRow.querySelector('td:nth-child(2)');
                    if (nameCell) personnelNameForSummary = nameCell.textContent.trim();
                } else {
                    // If no details in DOM yet, fetch one record to resolve name (first entry)
                    const quickResp = await fetch(`${API_BASE_URL}/api/after-hours${sParams.toString() ? `?${sParams.toString()}` : ''}`, { cache: 'no-store' });
                    if (quickResp.ok) {
                        const qj = await quickResp.json();
                        const ql = Array.isArray(qj?.data) ? qj.data : [];
                        if (ql.length > 0) {
                            personnelNameForSummary = ql[0].personnel_name || '';
                            // fallback: if no name, attempt personnel map
                            if (!personnelNameForSummary && ql[0].personnel_id) {
                                try {
                                    const pr = await fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' });
                                    if (pr.ok) {
                                        const pj = await pr.json();
                                        const p = (pj.data || []).find(pp => String(pp.id) === String(ql[0].personnel_id));
                                        if (p) personnelNameForSummary = p.name;
                                    }
                                } catch {}
                            }
                        }
                    }
                }
            } catch {}

            if (sumBody) {
                sumBody.innerHTML = `
                    <tr>
                        <td>${personnelNameForSummary || '-'}</td>
                        <td>${d.total_call_count || 0}</td>
                        <td>${d.total_talk_duration || 0}</td>
                        <td>${d.total_member_count || 0}</td>
                        <td>${d.record_count || 0}</td>
                    </tr>`;
            }
        } else {
            if (sumBody) sumBody.innerHTML = `<tr><td colspan="5" class="text-center text-danger">Özet yüklenemedi</td></tr>`;
        }
        // Details
        const dParams = new URLSearchParams();
        if (startDate) dParams.append('start_date', startDate);
        if (endDate) dParams.append('end_date', endDate);
        if (tableBody) tableBody.innerHTML = `<tr><td colspan="7" class="text-center">Yükleniyor...</td></tr>`;
        const listResp = await fetch(`${API_BASE_URL}/api/after-hours${dParams.toString() ? `?${dParams.toString()}` : ''}`, { cache: 'no-store' });
        if (listResp.ok) {
            const js = await listResp.json();
            const list = Array.isArray(js?.data) ? js.data : [];
            if (list.length === 0) {
                if (tableBody) tableBody.innerHTML = `<tr><td colspan="7" class="text-center">Kayıt yok</td></tr>`;
            } else {
                // Build personnel map to display names
                let pMap = {};
                try {
                    const pr = await fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' });
                    if (pr.ok) {
                        const pj = await pr.json();
                        (pj.data || []).forEach(p => { pMap[p.id] = p.name; });
                    }
                } catch {}
                const rows = list.map(r => `
                    <tr data-record-id="${r.id}">
                        <td>${r.date || ''}</td>
                        <td>${r.personnel_name || pMap[r.personnel_id] || `Personel ${r.personnel_id}`}</td>
                        <td>${r.call_count || 0}</td>
                        <td>${r.talk_duration || 0}</td>
                        <td>${r.member_count || 0}</td>
                        <td>${r.notes || '-'}</td>
                        <td>
                            <button class="btn-warning btn-sm edit-btn" data-record-id="${r.id}" title="Kaydı Düzenle" style="margin-right: 5px;"><i class="fas fa-edit"></i></button>
                            <button class="btn-danger btn-sm delete-btn" data-record-id="${r.id}" title="Kaydı Sil"><i class="fas fa-trash"></i></button>
                        </td>
                    </tr>`).join('');
                if (tableBody) tableBody.innerHTML = rows;
            }
        } else {
            if (tableBody) tableBody.innerHTML = `<tr><td colspan="7" class="text-center text-danger">Kayıtlar yüklenemedi</td></tr>`;
        }
    } catch (e) {
        console.error('❌ updateAfterHoursContent failed:', e);
        if (sumBody) sumBody.innerHTML = `<tr><td colspan="5" class="text-center text-danger">Özet yüklenemedi</td></tr>`;
        if (tableBody) tableBody.innerHTML = `<tr><td colspan="7" class="text-center text-danger">Kayıtlar yüklenemedi</td></tr>`;
    }
}

function clearAfterHoursFilter() {
    const s = document.getElementById('afterHoursStartDate');
    const e = document.getElementById('afterHoursEndDate');
    if (s) s.value = '';
    if (e) e.value = '';
    updateAfterHoursContent();
}

async function exportAfterHoursExcel() {
    try {
        const startDate = document.getElementById('afterHoursStartDate')?.value || '';
        const endDate = document.getElementById('afterHoursEndDate')?.value || '';
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        const url = `${API_BASE_URL}/api/export/after-hours-excel${params.toString() ? `?${params.toString()}` : ''}`;
        const resp = await fetch(url, { cache: 'no-store' });
        if (!resp.ok) {
            let msg = 'Bilinmeyen hata';
            try { const e = await resp.json(); msg = e.detail || msg; } catch {}
            showNotification(`❌ Excel export başarısız: ${msg}`, 'error');
            return;
        }
        const blob = await resp.blob();
        const dl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = dl;
        const today = new Date().toISOString().split('T')[0];
        a.download = `mesai_sonrasi_${today}.xlsx`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(dl);
        showNotification('✅ Excel dosyası indirildi', 'success');
    } catch (e) {
        console.error('❌ After Hours export error', e);
        showNotification('❌ Excel export sırasında hata oluştu', 'error');
    }
}

async function editAfterHoursRecord(recordId) {
    try {
        // There is no single GET endpoint; fetch list and find the record
        const resp = await fetch(`${API_BASE_URL}/api/after-hours`, { cache: 'no-store' });
        if (!resp.ok) throw new Error('Liste yüklenemedi');
        const js = await resp.json();
        const list = Array.isArray(js?.data) ? js.data : [];
        const r = list.find(x => String(x.id) === String(recordId));
        if (!r) throw new Error('Kayıt bulunamadı');
        await loadPersonnelOptions('after_hours_personnel_id');
        document.getElementById('afterHoursEditId').value = r.id;
        document.getElementById('after_hours_personnel_id').value = r.personnel_id || '';
        document.getElementById('after_hours_date').value = r.date || '';
        document.getElementById('after_hours_call_count').value = r.call_count || '';
        document.getElementById('after_hours_talk_duration').value = r.talk_duration || '';
        document.getElementById('after_hours_member_count').value = r.member_count || '';
        document.getElementById('after_hours_notes').value = r.notes || '';
        openModal('afterHoursModal');
    } catch (e) {
        console.error('❌ editAfterHoursRecord failed:', e);
        alert('Kayıt yüklenemedi');
    }
}

async function deleteAfterHoursRecord(recordId) {
    if (!confirm('Bu mesai sonrası kaydını silmek istediğinizden emin misiniz?')) return;
    try {
        // Optimistic remove
        const row = document.querySelector(`tr[data-record-id="${recordId}"]`);
        if (row) { row.classList.add('fade-out'); setTimeout(() => row.remove(), 150); }
        const resp = await fetch(`${API_BASE_URL}/api/after-hours/${recordId}`, { 
            method: 'DELETE'
        });
        if (!resp.ok) {
            let msg = 'Bilinmeyen hata';
            try { const e = await resp.json(); msg = e.detail || msg; } catch {}
            showNotification(`❌ Silme başarısız: ${msg}`, 'error');
            // reload list to recover if optimistic removal happened
            setTimeout(() => updateAfterHoursContent().catch(()=>{}), 200);
            return;
        }
        showNotification('✅ Kayıt silindi', 'success');
        setTimeout(() => updateAfterHoursContent().catch(()=>{}), 150);
    } catch (e) {
        console.error('❌ deleteAfterHoursRecord failed:', e);
        alert('Silme sırasında hata oluştu');
    }
}

function updateTargetsContent() {
    const container = document.getElementById('targetDistributionContainer');
    if (!container) return;
    container.innerHTML = `<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i><p>Hedef verileri yükleniyor...</p></div>`;
    (async () => {
        try {
            const [targetsResp, personnelResp] = await Promise.all([
                fetch(`${API_BASE_URL}/api/targets`, { cache: 'no-store' }),
                fetch(`${API_BASE_URL}/api/personnel`, { cache: 'no-store' })
            ]);
            if (!targetsResp.ok) throw new Error('Targets API not OK');
            const targetsJson = await targetsResp.json();
            const targets = Array.isArray(targetsJson?.data) ? targetsJson.data : [];
            let personnelMap = {};
            if (personnelResp.ok) {
                const pr = await personnelResp.json();
                (pr.data || []).forEach(p => { personnelMap[p.id] = p.name; });
            }
            if (targets.length === 0) {
                container.innerHTML = `<div class="text-center"><p>Henüz hedef kaydı yok</p></div>`;
                return;
            }
            // Compute achieved totals per target (by type) and remaining distance
            const typeFieldMap = {
                'uye_adedi': 'member_count',
                'whatsapp_adedi': 'whatsapp_count',
                'cihaz_adedi': 'device_count',
                'whatsapp_cevapsiz': 'unanswered_count'
            };

        const progressList = await Promise.all(targets.map(async (t) => {
                try {
                    const params = new URLSearchParams();
            // Use overall totals from Performance (no date filter)
            params.append('personnel_id', String(t.personnel_id));
                    const url = `${API_BASE_URL}/api/performance-records?${params.toString()}`;
                    const resp = await fetch(url, { cache: 'no-store' });
                    if (!resp.ok) throw new Error('Perf API not OK');
                    const js = await resp.json();
                    const list = Array.isArray(js?.data) ? js.data : [];
                    const field = typeFieldMap[t.target_type] || 'member_count';
                    const achieved = list.reduce((acc, r) => acc + (parseInt(r[field] || 0) || 0), 0);
                    const remaining = Math.max((parseInt(t.target_value || 0) || 0) - achieved, 0);
                    return { id: t.id, achieved, remaining };
                } catch (_) {
                    return { id: t.id, achieved: null, remaining: null };
                }
            }));

            const progMap = Object.fromEntries(progressList.map(p => [p.id, p]));

            const rows = targets.map(t => {
                const prog = progMap[t.id] || {};
                const remaining = (prog.remaining == null) ? '-' : String(prog.remaining);
                return `
                    <tr data-target-id="${t.id}">
                        <td>${personnelMap[t.personnel_id] || `Personel ${t.personnel_id}`}</td>
                        <td>${t.target_type || 'uye_adedi'}</td>
                        <td>${t.target_value}</td>
                        <td>${t.start_date} - ${t.end_date}</td>
                        <td>
                            <span class="badge ${prog.remaining === 0 ? 'badge-success' : 'badge-warning'}">${remaining}</span>
                        </td>
                        <td class="actions">
                            <button class="btn btn-warning btn-sm target-edit-btn" data-target-id="${t.id}"><i class="fas fa-edit"></i> Düzenle</button>
                            <button class="btn btn-danger btn-sm target-delete-btn" data-target-id="${t.id}"><i class="fas fa-trash"></i> Sil</button>
                        </td>
                    </tr>`;
            }).join('');

            container.innerHTML = `
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Personel</th>
                                <th>Hedef Tipi</th>
                                <th>Değer</th>
                                <th>Tarih Aralığı</th>
                                <th>Hedefe Kalan Uzaklık</th>
                                    <th>İşlemler</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>
            `;
        } catch (e) {
            console.error('❌ Targets load error:', e);
            container.innerHTML = `<div class="text-center text-danger">Hedef verileri yüklenemedi</div>`;
        }
    })();
}

// === TARGETS: Edit/Delete handlers ===
async function editTarget(targetId) {
    try {
        const resp = await fetch(`${API_BASE_URL}/api/targets`, { cache: 'no-store' });
        if (!resp.ok) throw new Error('Targets API not OK');
        const json = await resp.json();
        const t = (json.data || []).find(x => x.id === targetId);
        if (!t) throw new Error('Hedef bulunamadı');

        // Load personnel options then set values
        await loadPersonnelOptions('target_personnel_id');
        const form = document.getElementById('setTargetsForm');
        if (form && typeof form.reset === 'function') form.reset();
        const hid = document.getElementById('editTargetId');
        if (hid) hid.value = String(targetId);
        const sel = document.getElementById('target_personnel_id');
        if (sel) sel.value = String(t.personnel_id);
        const sDate = document.getElementById('target_start_date');
        const eDate = document.getElementById('target_end_date');
        const count = document.getElementById('target_member_count');
        if (sDate) sDate.value = t.start_date || '';
        if (eDate) eDate.value = t.end_date || '';
        if (count) count.value = t.target_value != null ? String(t.target_value) : '';

        openModal('setTargetsModal');
    } catch (e) {
        console.error('❌ Hedef düzenleme yükleme hatası:', e);
        showNotification('❌ Hedef verisi yüklenemedi', 'error');
    }
}

async function deleteTarget(targetId) {
    if (!confirm('Bu hedefi silmek istediğinizden emin misiniz?')) return;
    try {
        // Optimistic remove from DOM
        const row = document.querySelector(`tr[data-target-id="${targetId}"]`);
        if (row) { row.classList.add('fade-out'); setTimeout(() => row.remove(), 200); }
        const resp = await fetch(`${API_BASE_URL}/api/targets/${targetId}`, {
            method: 'DELETE'
        });
        if (!resp.ok) {
            let msg = 'Bilinmeyen hata';
            try { const j = await resp.json(); msg = j.detail || msg; } catch {}
            showNotification(`❌ Hedef silinemedi: ${msg}`, 'error');
            await updateTargetsContent();
            return;
        }
        showNotification('✅ Hedef silindi', 'success');
        setTimeout(() => { updateTargetsContent().catch(()=>{}); }, 150);
    } catch (e) {
        console.error('❌ Hedef silme hatası:', e);
        showNotification('❌ Bağlantı hatası', 'error');
        await updateTargetsContent();
    }
}

// MODAL FONKSİYONLARI
function openModal(modalId) {
    console.log(`📂 Opening modal: ${modalId}`);
    
    // ÖNCELİKLE TÜM MODALLARI KAPAT
    closeModals();
    
    const modal = document.getElementById(modalId);
    
    if (modal) {
        console.log('✅ Modal element found:', modal);
        
        // Parent modal-overlay'i bul ve göster
        const modalOverlay = modal.closest('.modal-overlay');
        if (modalOverlay) {
            console.log('✅ Modal overlay found, showing it');
            
            // Önce display'i ayarla
            modalOverlay.style.display = 'flex';
            modalOverlay.style.position = 'fixed';
            modalOverlay.style.top = '0';
            modalOverlay.style.left = '0';
            modalOverlay.style.width = '100%';
            modalOverlay.style.height = '100%';
            modalOverlay.style.zIndex = '999999';
            modalOverlay.style.backgroundColor = 'rgba(0,0,0,0.8)';
            modalOverlay.style.opacity = '1';
            modalOverlay.style.visibility = 'visible';
            modalOverlay.style.justifyContent = 'center';
            modalOverlay.style.alignItems = 'center';
            
            // Kısa bir gecikmeyle active sınıfını ekle (CSS transition için)
            setTimeout(() => {
                modalOverlay.classList.add('active');
                console.log('✅ Active class added to modal overlay');
                
                // DEBUG: Modal'ın gerçek stillerini kontrol et
                const computedStyle = window.getComputedStyle(modalOverlay);
                console.log('🔍 Modal computed styles:', {
                    display: computedStyle.display,
                    opacity: computedStyle.opacity,
                    visibility: computedStyle.visibility,
                    zIndex: computedStyle.zIndex,
                    position: computedStyle.position
                });
                console.log('🔍 Modal classList:', modalOverlay.classList.toString());
                console.log('🔍 Modal inline styles:', modalOverlay.style.cssText);
            }, 10);
            
            // CSS ile çakışmaması için inline style'ları daha da güçlü yap
            
            // Modal içeriğinin de görünür olduğundan emin ol
            modal.style.display = 'block';
            modal.style.position = 'relative';
            modal.style.zIndex = '999997';
            modal.style.margin = 'auto';
            modal.style.maxWidth = '600px';
            modal.style.width = '90%';
            
            // Modal'ın kendisini göster
            modal.style.display = 'block';
            modal.style.backgroundColor = 'white';
            modal.style.borderRadius = '8px';
            modal.style.boxShadow = '0 4px 20px rgba(0,0,0,0.8)';
            modal.style.maxWidth = '90vw';
            modal.style.maxHeight = '90vh';
            modal.style.overflow = 'auto';
            modal.style.padding = '20px';
            
            console.log('✅ Modal should now be visible');
            
            // Body'ye overflow hidden ekle
            document.body.style.overflow = 'hidden';
            
            console.log('✅ Modal styles applied:', {
                display: modal.style.display,
                visibility: modal.style.visibility,
                zIndex: modal.style.zIndex,
                position: modal.style.position
            });
            
            // Backdrop (modalOverlay) click ile kapat
            modalOverlay.onclick = (e) => {
                if (e.target === modalOverlay) { // Sadece overlay'e tıklanırsa
                    console.log('🖱️ Backdrop clicked');
                    closeModals();
                }
            };
            
            // Close button ile kapat
            const closeBtn = modal.querySelector('.modal-close');
            if (closeBtn) {
                closeBtn.onclick = () => {
                    console.log('🖱️ Close button clicked');
                    closeModals();
                };
            }
            
            document.body.classList.add('modal-open');
            
            // Personel gerektiren modallarda personel listesini yükle
            if (modalId === 'addPersonnelModal') {
                // Set today's date as default hire date
                const today = new Date().toISOString().split('T')[0];
                const hireDateField = document.getElementById('hire_date');
                if (hireDateField) {
                    hireDateField.value = today;
                }
            } else if (modalId === 'addPerformanceModal') {
                loadPersonnelOptions('performance_personnel_id');
            } else if (modalId === 'addTrainingFeedbackModal') {
                loadPersonnelOptions('training_feedback_personnel_id');
                populateTrainingFeedbackPersonnelDropdown(); // EXPLICIT CALL
                // Modal'ı normal ekleme modunda aç
                resetTrainingFeedbackModal();
            } else if (modalId === 'addRecordModal') {
                // Çağrı kayıt modalında personel seçeneklerini yükle
                console.log('🔄 Loading personnel for call record modal...');
                loadPersonnelOptions('personnel_id');
            } else if (modalId === 'setTargetsModal') {
                // Hedef belirleme modalı: add vs edit
                // Edit modunda (editTargetId doluysa) seçenekleri yeniden yükleyip seçimi sıfırlamayalım.
                try {
                    const hid = document.getElementById('editTargetId');
                    const sel = document.getElementById('target_personnel_id');
                    const isEdit = !!(hid && hid.value && String(hid.value).trim() !== '');
                    if (isEdit) {
                        // Eğer seçenekler zaten varsa, yeniden yüklemeye gerek yok.
                        if (sel && sel.options && sel.options.length > 0) {
                            // no-op
                        } else {
                            // Seçenekler boşsa yükle, mevcut değeri koru
                            const currentVal = sel ? sel.value : '';
                            loadPersonnelOptions('target_personnel_id')?.then?.(() => {
                                try { if (sel && currentVal) sel.value = currentVal; } catch {}
                            });
                        }
                    } else {
                        // Add modunda normal yükleme yap
                        loadPersonnelOptions('target_personnel_id');
                    }
                } catch {
                    // Her ihtimale karşı varsayılan yüklemeyi yap
                    loadPersonnelOptions('target_personnel_id');
                }
            }
            
            console.log(`✅ Modal opened: ${modalId}`);
        } else {
            console.error('❌ Modal overlay not found');
        }
    } else {
        console.error(`❌ Modal not found: ${modalId}`);
    showNotification(`Modal bulunamadı: ${modalId}`, 'error', 'Hata');
    }
}

function closeModals() {
    console.log('❌ Closing all modals - FORCE CLOSE');
    
    // TÜM MODAL OVERLAY'LERİ KAPAT
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.style.display = 'none';
        overlay.classList.remove('active'); // CRITICAL: active class'ını kaldır
        console.log('✅ Modal overlay closed');
    });
    
    // TÜM MODALLARI ZORLA KAPAT
    document.querySelectorAll('.modal').forEach(modal => {
        modal.style.display = 'none';
        modal.classList.remove('active');
        console.log('✅ Modal closed:', modal.id);
    });
    
    // Backdrop'u da kapat
    const backdrop = document.getElementById('modalBackdrop');
    if (backdrop) {
        backdrop.style.display = 'none';
    }
    
    // OVERLAY KAPATMA (eski sistem)
    const overlay = document.getElementById('modalOverlay');
    if (overlay) {
        overlay.style.display = 'none';
        overlay.classList.remove('active');
        console.log('✅ Overlay closed');
    }
    
    // BODY CLASS TEMİZLE VE OVERFLOW RESTORE ET
    document.body.classList.remove('modal-open');
    document.body.style.overflow = '';
    
    console.log('✅ All modals force closed');
}

// Make closeModals globally accessible
window.closeModals = closeModals;

// Delegated delete handler for Attendance Override
document.addEventListener('click', async (event) => {
    const btn = event.target.closest && event.target.closest('#deleteAttendanceOverrideBtn');
    if (!btn) return;
    event.preventDefault();
    const id = document.getElementById('attendance_override_edit_id')?.value;
    if (!id) { closeModals(); return; }
    if (!confirm('Bu puantaj kaydını silmek istediğinizden emin misiniz?')) return;
    try {
        const resp = await fetch(`${API_BASE_URL}/api/attendance/${id}`, {
            method: 'DELETE'
        });
        if (!resp.ok) {
            let msg = 'Bilinmeyen hata';
            try { const j = await resp.json(); msg = j.detail || msg; } catch {}
            showNotification(`❌ Silme başarısız: ${msg}`, 'error');
            return;
        }
        showNotification('✅ Kayıt silindi', 'success');
        closeModals();
        setTimeout(() => { updateAttendanceContent && updateAttendanceContent(); }, 120);
    } catch (e) {
        console.error('❌ Attendance override delete failed:', e);
        alert('Silme sırasında hata oluştu');
    }
});

// FORM SUBMIT FONKSİYONU - VALİDATION İLE
async function handleFormSubmit(form) {
    console.log('📝 Handling form submit:', form.id);
    
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    console.log('📋 Form data:', data);
    
    // FORM VALİDATION - SADECE BOŞ KONTROL
    if (form.id === 'addPersonnelForm') {
        if (!data.name || data.name.trim() === '') {
            console.log('⚠️ Personel adı boş, işleme devam edilmiyor');
            alert('⚠️ Personel adı zorunludur!');
            return;
        }
        if (!data.username || data.username.trim() === '') {
            console.log('⚠️ Kullanıcı adı boş, işleme devam edilmiyor');
            alert('⚠️ Kullanıcı adı zorunludur!');
            return;
        }
        if (!data.hire_date) {
            console.log('⚠️ İşe giriş tarihi boş, işleme devam edilmiyor');
            alert('⚠️ İşe giriş tarihi zorunludur!');
            return;
        }
        if (!data.team) {
            console.log('⚠️ Ekip seçimi yapılmamış, işleme devam edilmiyor');
            alert('⚠️ Ekip seçimi zorunludur!');
            return;
        }
        console.log('✅ Personel formu validation passed:', data);
    }
    
    if (form.id === 'editPersonnelForm') {
        if (!data.editName || data.editName.trim() === '') {
            console.log('⚠️ Personel adı boş, işleme devam edilmiyor');
            alert('⚠️ Personel adı zorunludur!');
            return;
        }
        if (!data.editUsername || data.editUsername.trim() === '') {
            console.log('⚠️ Kullanıcı adı boş, işleme devam edilmiyor');
            alert('⚠️ Kullanıcı adı zorunludur!');
            return;
        }
        if (!data.editHireDate) {
            console.log('⚠️ İşe giriş tarihi boş, işleme devam edilmiyor');
            alert('⚠️ İşe giriş tarihi zorunludur!');
            return;
        }
        if (!data.editTeam) {
            console.log('⚠️ Ekip seçimi yapılmamış, işleme devam edilmiyor');
            alert('⚠️ Ekip seçimi zorunludur!');
            return;
        }
        console.log('✅ Personel düzenleme formu validation passed:', data);
    }
    
    if (form.id === 'addRecordForm') {
        if (!data.record_date || !data.personnel_id || !data.call_number || !data.score) {
            console.log('⚠️ Kayıt formu eksik bilgi, işleme devam edilmiyor');
            return;
        }
        console.log('✅ Kayıt formu validation passed');
    }
    
    if (form.id === 'addPerformanceForm') {
        if (!data.performance_date || !data.performance_personnel_id) {
            console.log('⚠️ Performans formu eksik bilgi (tarih ve personel gerekli), işleme devam edilmiyor');
            alert('⚠️ Tarih ve personel seçimi zorunludur!');
            return;
        }
        console.log('✅ Performans formu validation passed');
    }
    
    if (form.id === 'addTrainingFeedbackForm') {
        if (!data.training_feedback_date || !data.training_feedback_personnel_id) {
            console.log('⚠️ Eğitim-geribildirim formu eksik bilgi (tarih ve personel gerekli), işleme devam edilmiyor');
            alert('⚠️ Tarih ve personel seçimi zorunludur!');
            return;
        }
        console.log('✅ Eğitim-geribildirim formu validation passed');
    }
    
    try {
        let response;
        
        switch(form.id) {
            case 'addPersonnelForm':
                response = await fetch(`${API_BASE_URL}/api/personnel`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name: data.name.trim(),
                        username: data.username.trim(),
                        email: data.email ? data.email.trim() : '',
                        reference: data.reference ? data.reference.trim() : '',
                        hire_date: data.hire_date,
                        team: data.team,
                        promotion_date: data.promotion_date || ''
                    })
                });
                break;
                
            case 'editPersonnelForm':
                const personnelId = data.editPersonnelId;
                response = await fetch(`${API_BASE_URL}/api/personnel/${personnelId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name: data.editName.trim(),
                        username: data.editUsername.trim(),
                        email: data.editEmail ? data.editEmail.trim() : '',
                        reference: data.editReference ? data.editReference.trim() : '',
                        hire_date: data.editHireDate,
                        team: data.editTeam,
                        promotion_date: data.editPromotionDate || ''
                    })
                });
                break;
                
            case 'addRecordForm':
                response = await fetch(`${API_BASE_URL}/api/daily-records`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        date: data.record_date,
                        personnel_id: data.personnel_id,
                        call_number: data.call_number,
                        score: parseInt(data.score),
                        notes: data.notes
                    })
                });
                break;
                
            case 'addPerformanceForm':
                response = await fetch(`${API_BASE_URL}/api/performance-records`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        performance_date: data.performance_date,
                        performance_personnel_id: data.performance_personnel_id,
                        member_count: data.member_count || 0,
                        whatsapp_count: data.whatsapp_count || 0,
                        device_count: data.device_count || 0,
                        unanswered_count: data.unanswered_count || 0,
                        knowledge_duel_result: data.knowledge_duel_result || 0,
                        reward_penalty: data.reward_penalty || "",
                        performance_notes: data.performance_notes || ""
                    })
                });
                break;
                
            case 'addTrainingFeedbackForm':
                console.log('🔍 Training feedback form data before mapping:', data);
                
                // Edit mode kontrolü
                const editRecordId = data.editRecordId;
                if (editRecordId) {
                    // Güncelleme işlemi
                    console.log('🔄 Güncelleme işlemi için PUT request:', editRecordId);
                    response = await fetch(`${API_BASE_URL}/api/training-feedback/${editRecordId}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            personnel_id: parseInt(data.training_feedback_personnel_id),
                            date: data.training_feedback_date,
                            feedback_count: parseInt(data.feedback_count) || 0,
                            feedback_subject: data.feedback_subject || "",
                            general_training_count: parseInt(data.general_training_count) || 0,
                            general_training_subject: data.general_training_subject || "",
                            one_on_one_training_count: parseInt(data.one_on_one_training_count) || 0,
                            one_on_one_training_subject: data.one_on_one_training_subject || "",
                            notes: data.notes || ""
                        })
                    });
                } else {
                    // Yeni kayıt ekleme işlemi
                    console.log('➕ Yeni kayıt ekleme işlemi');
                    response = await fetch(`${API_BASE_URL}/api/training-feedback`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            personnel_id: parseInt(data.training_feedback_personnel_id),
                            date: data.training_feedback_date,
                            feedback_count: parseInt(data.feedback_count) || 0,
                            feedback_subject: data.feedback_subject || "",
                            general_training_count: parseInt(data.general_training_count) || 0,
                            general_training_subject: data.general_training_subject || "",
                            one_on_one_training_count: parseInt(data.one_on_one_training_count) || 0,
                            one_on_one_training_subject: data.one_on_one_training_subject || "",
                            notes: data.notes || ""
                        })
                    });
                }
                break;
                
            case 'editTrainingFeedbackForm':
                console.log('🔄 Edit Training Feedback Form data:', data);
                const editTFRecordId = data.id; // Hidden input'tan gelen ID
                response = await fetch(`${API_BASE_URL}/api/training-feedback/${editTFRecordId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        personnel_id: parseInt(data.training_feedback_personnel_id),
                        date: data.training_feedback_date,
                        feedback_count: parseInt(data.feedback_count) || 0,
                        feedback_subject: data.feedback_subject || "",
                        general_training_count: parseInt(data.general_training_count) || 0,
                        general_training_subject: data.general_training_subject || "",
                        one_on_one_training_count: parseInt(data.one_on_one_training_count) || 0,
                        one_on_one_training_subject: data.one_on_one_training_subject || "",
                        notes: data.notes || ""
                    })
                });
                break;
                
            case 'editPerformanceForm':
                const recordId = data.editPerformanceId;
                response = await fetch(`${API_BASE_URL}/api/performance-records/${recordId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        performance_date: data.editPerformanceDate,
                        performance_personnel_id: data.editPerformancePersonnel,
                        member_count: data.editPerformanceMemberCount || 0,
                        whatsapp_count: data.editPerformanceWhatsappCount || 0,
                        device_count: data.editPerformanceDeviceCount || 0,
                        unanswered_count: data.editPerformanceUnansweredCount || 0,
                        knowledge_duel_result: data.editPerformanceKnowledgeDuelResult || 0,
                        reward_penalty: data.editPerformanceRewardPenalty || ""
                    })
                });
                break;
                
            case 'exportExcelForm':
                if (!data.export_start_date || !data.export_end_date) {
                    alert('⚠️ Başlangıç ve bitiş tarihlerini seçiniz!');
                    return;
                }
                window.open(`${API_BASE_URL}/api/export/excel?start_date=${data.export_start_date}&end_date=${data.export_end_date}`);
                closeModals();
                return;
                
            case 'setTargetsForm':
                if (!data.target_personnel_id || !data.target_start_date || !data.target_end_date || !data.target_member_count) {
                    alert('⚠️ Tüm alanları doldurmanız gerekiyor!');
                    return;
                }
                response = await fetch(`${API_BASE_URL}/api/targets`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        personnel_id: data.target_personnel_id,
                        start_date: data.target_start_date,
                        end_date: data.target_end_date,
                        member_count: parseInt(data.target_member_count)
                    })
                });
                break;
                
            default:
                console.log('❌ Unknown form:', form.id);
                return;
        }
        
        if (response && response.ok) {
            const result = await response.json();
            console.log('✅ Success:', result);
            
            // Form tipine göre doğru success mesajı
            if (form.id === 'editTrainingFeedbackForm') {
                toastManager.success('Eğitim geri bildirimi başarıyla güncellendi!');
            } else if (form.id === 'editPerformanceForm') {
                toastManager.success('Performans kaydı başarıyla güncellendi!');
            } else if (form.id === 'editPersonnelForm') {
                toastManager.success('Personel bilgileri başarıyla güncellendi!');
            } else if (form.id.startsWith('edit')) {
                toastManager.success('Kayıt başarıyla güncellendi!');
            } else {
                toastManager.success('Kayıt başarıyla eklendi!');
            }
            
            // MODAL KAPAMA
            console.log('🚪 Closing modals...');
            closeModals();
            
            // FORM TEMİZLEME VE RESET
            console.log('🧹 Resetting form...');
            form.reset();
            
            // Training feedback modal'ı resetle
            if (form.id === 'addTrainingFeedbackForm') {
                resetTrainingFeedbackModal();
            }
            
            // SAYFA YENİLEME
            const currentPage = document.querySelector('.page.active').id.replace('-page', '');
            console.log('🔄 Updating page content:', currentPage);
            await updatePageContent(currentPage);
            
        } else if (response) {
            console.log('❌ Response not OK. Status:', response.status, 'StatusText:', response.statusText);
            const error = await response.json();
            console.error('❌ Error details:', error);
            alert('❌ Hata: ' + (error.detail || 'İşlem başarısız!'));
        }
        
    } catch (error) {
        console.error('❌ Network error:', error);
        alert('❌ Bağlantı hatası! Backend sunucusu çalışıyor mu?');
    }
}

// Personel silme fonksiyonu
// === PERSONEL DÜZENLEME FONKSİYONLARI ===
async function editPersonnel(personnelId) {
    console.log(`✏️ Editing personnel with ID: ${personnelId}`);
    
    try {
        // Personel bilgilerini getir
        const response = await fetch(`${API_BASE_URL}/api/personnel/${personnelId}`);
        if (response.ok) {
            const result = await response.json();
            const personnel = result.data;
            
            // Formu doldur
            document.getElementById('editPersonnelId').value = personnel.id;
            document.getElementById('editName').value = personnel.name || '';
            document.getElementById('editUsername').value = personnel.username || '';
            document.getElementById('editEmail').value = personnel.email || '';
            document.getElementById('editReference').value = personnel.reference || '';
            document.getElementById('editHireDate').value = personnel.hire_date || '';
            document.getElementById('editTeam').value = personnel.team || '';
            document.getElementById('editPromotionDate').value = personnel.promotion_date || '';
            
            // Modalı aç
            openModal('editPersonnelModal');
            // Artık manuel submit handler eklemiyoruz; global capture + handleFormSubmitInline devrede.
            console.log('🧪 editPersonnel: Using unified inline submit handler (no manual listener).');
        } else {
            throw new Error('Personel bilgileri alınamadı');
        }
    } catch (error) {
        console.error('❌ Error loading personnel for edit:', error);
        alert('Personel bilgileri yüklenirken hata oluştu: ' + error.message);
    }
}

async function deletePersonnel(personnelId) {
    if (!confirm('Bu personeli silmek istediğinizden emin misiniz?')) {
        return;
    }
    
    try {
        console.log(`🗑️ Deleting personnel with ID: ${personnelId}`);
        
        const response = await fetch(`${API_BASE_URL}/api/personnel/${personnelId}`, {
            method: 'DELETE',
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Personnel deleted:', result);
            
            // Sayfa içeriğini güncelle
            await updatePersonnelContent();
            
            console.log('✅ Personel başarıyla silindi!');
        } else {
            const error = await response.json();
            console.error('❌ Delete error:', error);
            alert('❌ Hata: ' + (error.detail || 'Personel silinemedi!'));
        }
        
    } catch (error) {
        console.error('❌ Network error during delete:', error);
        alert('❌ Bağlantı hatası! Backend sunucusu çalışıyor mu?');
    }
}

// Personel seçeneklerini yükle
async function loadPersonnelOptions(selectId) {
    const selectElement = document.getElementById(selectId);
    if (!selectElement) {
        console.error(`❌ Select element not found: ${selectId}`);
        return;
    }
    
    try {
        console.log('🔄 Loading personnel options...');
        const response = await fetch(`${API_BASE_URL}/api/personnel`);
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Personnel options loaded:', result.data);
            
            // Mevcut seçenekleri temizle (ilk option hariç)
            selectElement.innerHTML = '<option value="">Personel Seçin</option>';
            
            // Personelleri ekle
            if (result.data && result.data.length > 0) {
                result.data.forEach(person => {
                    const option = document.createElement('option');
                    option.value = person.id;
                    option.textContent = person.name;
                    selectElement.appendChild(option);
                });
                console.log(`✅ ${result.data.length} personnel options added to ${selectId}`);
            } else {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'Personel bulunamadı';
                option.disabled = true;
                selectElement.appendChild(option);
            }
        } else {
            throw new Error('API response not OK');
        }
    } catch (error) {
        console.error('❌ Error loading personnel options:', error);
        selectElement.innerHTML = '<option value="">Personel yüklenemedi</option>';
    }
}

// Performans kaydı silme fonksiyonu
async function deletePerformanceRecord(recordId) {
    if (!confirm('Bu performans kaydını silmek istediğinizden emin misiniz?')) {
        return;
    }
    
    try {
        console.log(`🗑️ Deleting performance record: ${recordId}`);
        // Optimistic UI: remove the row immediately
        const row = document.querySelector(`tr[data-record-id="${recordId}"]`);
        if (row && row.parentElement) {
            row.classList.add('fade-out');
            setTimeout(() => row.remove(), 200);
        }
        const response = await fetch(`${API_BASE_URL}/api/performance-records/${recordId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Performance record deleted:', result);
            showNotification('✅ Performans kaydı başarıyla silindi!', 'success');
            // Fast refresh to keep summary in sync
            setTimeout(() => { updatePerformanceContent().catch(()=>{}); }, 150);
        } else {
            const errorResult = await response.json();
            console.error('❌ Delete failed:', errorResult);
            alert(`❌ Silme işlemi başarısız: ${errorResult.detail || 'Bilinmeyen hata'}`);
            // Rebuild table in case optimistic removal happened
            await updatePerformanceContent();
        }
    } catch (error) {
        console.error('❌ Error deleting performance record:', error);
        alert('❌ Bağlantı hatası! Backend sunucusu çalışıyor mu?');
        // Attempt to restore state
        await updatePerformanceContent();
    }
}

// Performans kaydı düzenleme fonksiyonu
async function editPerformanceRecord(recordId) {
    try {
        console.log(`✏️ Loading performance record for edit: ${recordId}`);
        const response = await fetch(`${API_BASE_URL}/api/performance-records/${recordId}`);
        
        if (response.ok) {
            const result = await response.json();
            const record = result.data;
            
            // Form alanlarını doldur
            document.getElementById('editPerformanceId').value = record.id;
            document.getElementById('editPerformancePersonnel').value = record.personnel_id;
            document.getElementById('editPerformanceDate').value = record.date;
            document.getElementById('editPerformanceMemberCount').value = record.member_count || '';
            document.getElementById('editPerformanceWhatsappCount').value = record.whatsapp_count || '';
            document.getElementById('editPerformanceDeviceCount').value = record.device_count || '';
            document.getElementById('editPerformanceUnansweredCount').value = record.unanswered_count || '';
            document.getElementById('editPerformanceKnowledgeDuelResult').value = record.knowledge_duel_result || '';
            document.getElementById('editPerformanceRewardPenalty').value = record.reward_penalty || '';
            
            // Personel seçeneklerini yükle
            await loadPersonnelOptions('editPerformancePersonnel');
            document.getElementById('editPerformancePersonnel').value = record.personnel_id;
            
            // Modal'ı aç
            openModal('editPerformanceModal');
        } else {
            throw new Error('Kayıt bulunamadı');
        }
    } catch (error) {
        console.error('❌ Error loading performance record:', error);
        alert('❌ Performans kaydı yüklenemedi!');
    }
}

// Çağrı kaydı silme fonksiyonu
async function deleteRecord(recordId) {
    if (!confirm('Bu çağrı kaydını silmek istediğinizden emin misiniz?')) {
        return;
    }
    
    try {
        console.log(`🗑️ Deleting daily record: ${recordId}`);
        const response = await fetch(`${API_BASE_URL}/api/daily-records/${recordId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Daily record deleted:', result);
            console.log('✅ Çağrı kaydı başarıyla silindi!');
            
            // Tabloyu güncelle
            await updateRecordsContent();
        } else {
            const errorResult = await response.json();
            console.error('❌ Delete failed:', errorResult);
            alert(`❌ Silme işlemi başarısız: ${errorResult.detail || 'Bilinmeyen hata'}`);
        }
    } catch (error) {
        console.error('❌ Error deleting daily record:', error);
        alert('❌ Bağlantı hatası! Backend sunucusu çalışıyor mu?');
    }
}

// 📊 EXCEL EXPORT FUNCTIONS
async function exportPersonnelToExcel() {
    try {
        console.log('📊 Starting personnel Excel export...');
        
        const response = await fetch(`${API_BASE_URL}/api/export/personnel-excel`, {
            method: 'GET'
        });
        
        if (response.ok) {
            // Blob olarak al
            const blob = await response.blob();
            
            // İndirme linki oluştur
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            
            // Dosya adı - tarih ile
            const today = new Date().toISOString().split('T')[0];
            a.download = `personel_listesi_${today}.xlsx`;
            
            // Linki tıkla ve temizle
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            console.log('✅ Personnel Excel export completed!');
            
            // Başarı mesajı göster
            showNotification('✅ Excel dosyası başarıyla indirildi!', 'success');
            
        } else {
            const errorResult = await response.json();
            console.error('❌ Export failed:', errorResult);
            alert(`❌ Excel export başarısız: ${errorResult.detail || 'Bilinmeyen hata'}`);
        }
        
    } catch (error) {
        console.error('❌ Error exporting personnel to Excel:', error);
        alert('❌ Excel export sırasında hata oluştu! Backend sunucusu çalışıyor mu?');
    }
}

// Records (Çağrı Puanları) için Excel export - mevcut sayfa filtreleri ile
let __exportRecordsBusy = false;
async function exportRecordsExcel() {
    if (__exportRecordsBusy) return;
    __exportRecordsBusy = true;
    const btn = document.getElementById('exportExcelBtn');
    if (btn) btn.disabled = true;
    try {
        const startDate = document.getElementById('recordsStartDate')?.value || '';
        const endDate = document.getElementById('recordsEndDate')?.value || '';
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        const url = `${API_BASE_URL}/api/export/excel${params.toString() ? `?${params.toString()}` : ''}`;
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) {
            let msg = 'Bilinmeyen hata';
            try { const e = await response.json(); msg = e.detail || msg; } catch {}
            showNotification(`❌ Excel export başarısız: ${msg}`, 'error');
            return;
        }
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        const today = new Date().toISOString().split('T')[0];
        a.download = `cagri_kayitlari_${today}.xlsx`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(downloadUrl);
        showNotification('✅ Excel dosyası indirildi', 'success');
    } catch (e) {
        console.error('❌ Records export error', e);
        showNotification('❌ Excel export sırasında hata oluştu', 'error');
    } finally {
        __exportRecordsBusy = false;
        if (btn) btn.disabled = false;
    }
}

// Notification gösterme için güvenli fallback (mevcut toast tabanlı showNotification'ı override ETME)
try {
    if (typeof window !== 'undefined' && typeof window.showNotification !== 'function') {
        window.showNotification = function(message, type = 'info', title = '') {
            // Toast sistemi hazır değilse basit console fallback
            const icon = type === 'success' ? '✅' : (type === 'error' ? '❌' : 'ℹ️');
            const prefix = title ? `${title}: ` : '';
            console.log(`${icon} ${prefix}${message}`);
        };
    }
} catch {}

// === PERFORMANS TARIH FİLTRELEME FONKSİYONLARI ===

// Performans tarih filtresi uygula
async function applyPerformanceDateFilter() {
    console.log('📅 Applying performance date filter...');
    
    const startDate = document.getElementById('performanceStartDateFilter').value;
    const endDate = document.getElementById('performanceEndDateFilter').value;
    const personnelId = document.getElementById('performancePersonnelFilter').value;
    
    console.log('🔍 Filter values:', {
        startDate, 
        endDate, 
        personnelId
    });
    
    // En az bir filtre değeri olmalı
    if (!startDate && !endDate && !personnelId) {
        alert('⚠️ Lütfen en az bir filtre değeri seçin!');
        return;
    }
    
    // Tarih validasyonu
    if (startDate && endDate && startDate > endDate) {
        alert('⚠️ Başlangıç tarihi bitiş tarihinden büyük olamaz!');
        return;
    }
    
    try {
        // URL parametrelerini oluştur
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        if (personnelId) params.append('personnel_id', personnelId);
        
        const url = `${API_BASE_URL}/api/performance-records?${params.toString()}`;
        console.log('🌐 API URL:', url);
        
        const response = await fetch(url);
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Filtered performance data loaded:', result);
            
            // Filtrelenmiş veriyi göster
            displayFilteredPerformanceData(result.data);
            
            // Filtre bilgisini göster
            showFilterInfo(result.filters);
            
        } else {
            throw new Error('API response not OK');
        }
        
    } catch (error) {
        console.error('❌ Error applying performance date filter:', error);
        alert('❌ Filtre uygulanırken hata oluştu! Backend sunucusu çalışıyor mu?');
    }
}

// Performans tarih filtresi temizle
async function clearPerformanceDateFilter() {
    console.log('🗑️ Clearing performance date filter...');
    
    // Input'ları temizle
    document.getElementById('performanceStartDateFilter').value = '';
    document.getElementById('performanceEndDateFilter').value = '';
    document.getElementById('performancePersonnelFilter').value = '';
    
    // Tüm veriyi yeniden yükle
    await updatePerformanceContent();
    
    // Filtre bilgisini gizle
    hideFilterInfo();
    
    console.log('✅ Performance date filter cleared');
}

// Filtrelenmiş performans verilerini göster
function displayFilteredPerformanceData(filteredData) {
    console.log('📊 Displaying filtered performance data:', filteredData);
    
    // Detaylı kayıtlar tablosunu güncelle
    const detailTableBody = document.querySelector('#performanceTableBody');
    if (detailTableBody) {
        if (filteredData && filteredData.length > 0) {
            detailTableBody.innerHTML = filteredData.map(record => {
                return `
                    <tr>
                        <td>${record.date}</td>
                        <td>${record.personnel_name || 'Personel ' + record.personnel_id}</td>
                        <td>${record.member_count || 0}</td>
                        <td>${record.whatsapp_count || 0}</td>
                        <td>${record.device_count || 0}</td>
                        <td>${record.unanswered_count || 0}</td>
                        <td>${record.knowledge_duel_result || 0}</td>
                        <td>${(record.reward_penalty || '').toString().toLowerCase() === 'odul' ? 'ödül' : '-'}</td>
                        <td>${(record.reward_penalty || '').toString().toLowerCase() === 'ceza' ? 'ceza' : '-'}</td>
                        <td>
                            <button class="btn-warning btn-sm edit-btn" data-record-id="${record.id}" title="Kaydı Düzenle" style="margin-right: 5px;">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="btn-danger btn-sm delete-btn" data-record-id="${record.id}" title="Kaydı Sil">
                                <i class="fas fa-trash"></i>
                            </button>
                        </td>
                    </tr>
                `;
            }).join('');
        } else {
        detailTableBody.innerHTML = `
                <tr>
            <td colspan="10" class="text-center">Filtre kriterlerine uygun performans kaydı bulunamadı</td>
                </tr>
            `;
        }
    }
    
    // Özet tablosunu güncelle
    const summaryTableBody = document.querySelector('#performanceSummaryTableBody');
    if (summaryTableBody) {
        if (filteredData && filteredData.length > 0) {
            // Personel bazında özet hesapla
            const summaryByPersonnel = {};
            
            filteredData.forEach(record => {
                const personnelId = record.personnel_id;
                if (!summaryByPersonnel[personnelId]) {
                    summaryByPersonnel[personnelId] = {
                        personnel_id: personnelId,
                        personnel_name: record.personnel_name || 'Personel ' + personnelId,
                        member_count: 0,
                        whatsapp_count: 0,
                        device_count: 0,
                        unanswered_count: 0,
                        knowledge_duel_result: 0,
                        reward_count: 0,
                        penalty_count: 0,
                        record_count: 0
                    };
                }
                
                summaryByPersonnel[personnelId].member_count += Number(record.member_count || 0);
                summaryByPersonnel[personnelId].whatsapp_count += Number(record.whatsapp_count || 0);
                summaryByPersonnel[personnelId].device_count += Number(record.device_count || 0);
                summaryByPersonnel[personnelId].unanswered_count += Number(record.unanswered_count || 0);
                summaryByPersonnel[personnelId].knowledge_duel_result += Number(record.knowledge_duel_result || 0);
                const rp = (record.reward_penalty || '').toString().toLowerCase();
                if (rp === 'odul') summaryByPersonnel[personnelId].reward_count += 1;
                else if (rp === 'ceza') summaryByPersonnel[personnelId].penalty_count += 1;
                summaryByPersonnel[personnelId].record_count += 1;
            });
            
            // Özet tablosunu oluştur
            summaryTableBody.innerHTML = Object.values(summaryByPersonnel).map(summary => {
                const kdTotal = Number(summary.knowledge_duel_result || 0);
                
                return `
                    <tr>
                        <td>${summary.personnel_name}</td>
                        <td>${summary.member_count}</td>
                        <td>${summary.whatsapp_count}</td>
                        <td>${summary.device_count}</td>
                        <td>${summary.unanswered_count}</td>
                        <td>${kdTotal}</td>
                        <td>${summary.reward_count || 0}</td>
                        <td>${summary.penalty_count || 0}</td>
                    </tr>
                `;
            }).join('');
        } else {
            summaryTableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">Filtre kriterlerine uygun performans özeti bulunamadı</td>
                </tr>
            `;
        }
    }
}

// Filtre bilgisi göster
function showFilterInfo(filters) {
    console.log('📋 Showing filter info:', filters);
    
    // Varolan filtre bilgi kutusunu bul veya oluştur
    let filterInfo = document.getElementById('performanceFilterInfo');
    if (!filterInfo) {
        filterInfo = document.createElement('div');
        filterInfo.id = 'performanceFilterInfo';
        filterInfo.className = 'filter-info-box';
        
        // Performans sayfasının başlığından sonra ekle
        const performanceHeader = document.querySelector('#performance-page .page-header');
        if (performanceHeader) {
            performanceHeader.insertAdjacentElement('afterend', filterInfo);
        }
    }
    
    // Filtre bilgisini oluştur
    let filterText = '🔍 Aktif Filtreler: ';
    const filterParts = [];
    
    if (filters.start_date) {
        filterParts.push(`Başlangıç: ${filters.start_date}`);
    }
    if (filters.end_date) {
        filterParts.push(`Bitiş: ${filters.end_date}`);
    }
    if (filters.personnel_id) {
        filterParts.push(`Personel ID: ${filters.personnel_id}`);
    }
    
    filterText += filterParts.join(' | ');
    filterText += ` | Sonuç: ${filters.total_filtered}/${filters.total_available} kayıt`;
    
    filterInfo.innerHTML = `
        <div class="filter-info-content">
            <i class="fas fa-info-circle"></i>
            <span>${filterText}</span>
            <button class="btn btn-sm btn-secondary" onclick="clearPerformanceDateFilter()">
                <i class="fas fa-times"></i> Filtreyi Temizle
            </button>
        </div>
    `;
    
    filterInfo.style.display = 'block';
}

// Filtre bilgisini gizle
function hideFilterInfo() {
    const filterInfo = document.getElementById('performanceFilterInfo');
    if (filterInfo) {
        filterInfo.style.display = 'none';
    }
}

// Filtre bilgi kutusu için CSS ekleyelim (dinamik olarak)
if (!document.getElementById('filterInfoStyles')) {
    const style = document.createElement('style');
    style.id = 'filterInfoStyles';
    style.textContent = `
        .filter-info-box {
            margin: 15px 0;
            padding: 12px 16px;
            background: rgba(33, 150, 243, 0.1);
            border: 1px solid rgba(33, 150, 243, 0.3);
            border-radius: 8px;
            backdrop-filter: blur(10px);
            display: none;
        }
        
        .filter-info-content {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #1976d2;
            font-weight: 500;
        }
        
        .filter-info-content i {
            color: #2196f3;
        }
        
        .filter-info-content .btn {
            margin-left: auto;
            padding: 4px 8px;
            font-size: 0.8em;
        }
    `;
    document.head.appendChild(style);
}

// === GLOBAL FONKSİYONLAR (HTML'den çağrılabilir) ===
window.editPersonnel = editPersonnel;
window.deletePersonnel = deletePersonnel;
window.exportPersonnelToExcel = exportPersonnelToExcel;
window.exportPerformanceToExcel = exportPerformanceToExcel;

// === PERFORMANS EXCEL EXPORT FONKSİYONU ===
let __exportPerformanceBusy = false;
async function exportPerformanceToExcel() {
    if (__exportPerformanceBusy) {
        console.log('⏳ Export already in progress, ignoring duplicate click');
        return;
    }
    __exportPerformanceBusy = true;
    console.log('📊 Starting performance Excel export...');
    
    try {
        // Mevcut filtreleri al
        const startDate = document.getElementById('performanceStartDateFilter').value;
        const endDate = document.getElementById('performanceEndDateFilter').value;
        const personnelId = document.getElementById('performancePersonnelFilter').value;
        
        console.log('🔍 Export filters:', {
            startDate, 
            endDate, 
            personnelId
        });
        
        // URL parametrelerini oluştur
        const params = new URLSearchParams();
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        if (personnelId) params.append('personnel_id', personnelId);
        
        const url = `${API_BASE_URL}/api/export/performance-excel?${params.toString()}`;
        console.log('🌐 Performance Excel Export URL:', url);
        
    const exportBtn = document.getElementById('exportPerformanceExcelBtn');
    if (exportBtn) exportBtn.disabled = true;
    const response = await fetch(url, { cache: 'no-store' });
        
        if (response.ok) {
            // Blob olarak al
            const blob = await response.blob();
            
            // Dosya adını oluştur
            let filename = 'performans_raporu';
            if (startDate || endDate) {
                filename += '_';
                if (startDate) filename += startDate;
                if (startDate && endDate) filename += '_to_';
                if (endDate) filename += endDate;
            }
            if (personnelId) {
                const personnelSelect = document.getElementById('performancePersonnelFilter');
                const selectedOption = personnelSelect.options[personnelSelect.selectedIndex];
                const personnelName = selectedOption.text.replace(/[^a-zA-Z0-9]/g, '_');
                filename += '_' + personnelName;
            }
            filename += '_' + new Date().toISOString().slice(0, 10).replace(/-/g, '') + '.xlsx';
            
            // Download link oluştur
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            
            // Otomatik indirme başlat
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            // Memory temizliği
            window.URL.revokeObjectURL(downloadUrl);
            
            console.log('✅ Performance Excel export completed!');
            
            // Başarı mesajı göster
            showNotification('✅ Performans Excel dosyası başarıyla indirildi!', 'success');
            
        } else {
            let errorMsg = 'Bilinmeyen hata';
            try {
                const errorResult = await response.json();
                console.error('❌ Performance export failed:', errorResult);
                errorMsg = errorResult.detail || errorMsg;
            } catch (_) {}
            showNotification(`❌ Performans Excel export başarısız: ${errorMsg}`, 'error');
        }
        
    } catch (error) {
        console.error('❌ Error exporting performance to Excel:', error);
        showNotification('❌ Performans Excel export sırasında hata oluştu! Backend sunucusu çalışıyor mu?', 'error');
    }
    finally {
        __exportPerformanceBusy = false;
        const exportBtn = document.getElementById('exportPerformanceExcelBtn');
        if (exportBtn) exportBtn.disabled = false;
    }
}

// Training-Feedback supporting functions
async function exportTrainingFeedbackExcel() {
    try {
        console.log('📊 Exporting training-feedback to Excel...');
        
        const startDate = document.getElementById('trainingFeedbackStartDate').value;
        const endDate = document.getElementById('trainingFeedbackEndDate').value;
        
        if (!startDate || !endDate) {
            alert('⚠️ Lütfen başlangıç ve bitiş tarihlerini seçin!');
            return;
        }
        
    const response = await fetch(`${API_BASE_URL}/api/export/training-feedback-excel?start_date=${startDate}&end_date=${endDate}`, {
            method: 'GET',
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `egitimler_raporu_${startDate}_to_${endDate}_${new Date().getTime()}.xlsx`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            
            console.log('✅ Training-feedback Excel export successful');
            showNotification('✅ Eğitimler Excel dosyası başarıyla indirildi!', 'success');
            
        } else {
            const errorResult = await response.json();
            console.error('❌ Training-feedback export failed:', errorResult);
            alert(`❌ Eğitimler Excel export başarısız: ${errorResult.detail || 'Bilinmeyen hata'}`);
        }
        
    } catch (error) {
        console.error('❌ Error exporting training-feedback to Excel:', error);
    alert('❌ Eğitimler Excel export sırasında hata oluştu! Backend sunucusu çalışıyor mu?');
    }
}

function showTrainingFeedbackBreakdown(personnelId, type) {
    console.log(`🔍 Showing training-feedback breakdown for personnel ${personnelId}, type: ${type}`);
    
    // Önce mevcut breakdown satırlarını kaldır
    document.querySelectorAll('.breakdown-row').forEach(row => row.remove());
    
    try {
        // Tüm kayıtları al ve personel için filtrele
    const startDate = document.getElementById('trainingFeedbackStartDate')?.value || '';
    const endDate = document.getElementById('trainingFeedbackEndDate')?.value || '';
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    fetch(`${API_BASE_URL}/api/training-feedback${params.toString() ? `?${params.toString()}` : ''}`)
            .then(response => response.json())
            .then(result => {
                if (result.success && result.data) {
                    // Personel için kayıtları filtrele
                    const personnelRecords = result.data.filter(record => record.personnel_id === personnelId);
                    
                    if (personnelRecords.length === 0) {
                        console.log(`Bu personel için ${type} kaydı bulunamadı.`);
                        return;
                    }
                    
                    // Type'a göre detayları hazırla
                    let details = [];
                    let title = '';
                    
            switch(type) {
            case 'feedback':
                            title = 'Geribildirim Detayları';
                            personnelRecords.forEach(record => {
                                if (record.feedback_count > 0 || record.feedback_topic) {
                                    details.push({
                                        date: record.date,
                                        subject: 'Geribildirim',
                    topic: record.feedback_subject || record.feedback_topic || 'Detay yok',
                                        count: record.feedback_count || 0
                                    });
                                }
                            });
                            break;
                        case 'general_training':
                            title = 'Genel Eğitim Detayları';
                            personnelRecords.forEach(record => {
                if (record.general_training_count > 0 || record.general_training_subject || record.general_training_topic) {
                                    details.push({
                                        date: record.date,
                                        subject: 'Genel Eğitim',
                    topic: record.general_training_subject || record.general_training_topic || 'Detay yok',
                                        count: record.general_training_count || 0
                                    });
                                }
                            });
                            break;
            case 'personal_training':
                            title = 'Birebir Eğitim Detayları';
                            personnelRecords.forEach(record => {
                const count = record.personal_training_count || record.individual_training_count;
                const topic = record.personal_training_subject || record.individual_training_topic;
                if (count > 0 || topic) {
                                    details.push({
                                        date: record.date,
                                        subject: 'Birebir Eğitim',
                    topic: topic || 'Detay yok',
                    count: count || 0
                                    });
                                }
                            });
                            break;
                    }
                    
                    if (details.length === 0) {
                        console.log(`Bu personel için ${title} bulunamadı.`);
                        return;
                    }
                    
                    // Personelin satırını bul
                    const summaryTable = document.querySelector('#trainingFeedbackSummaryTBody');
                    const rows = summaryTable.querySelectorAll('tr');
                    let targetRow = null;
                    
                    rows.forEach(row => {
                        const cells = row.querySelectorAll('td');
                        if (cells.length > 0) {
                            const personnelName = cells[0].textContent.trim();
                            // Bu satırın doğru personel satırı olup olmadığını kontrol et
                            // onclick attribute'dan personnel_id'yi çıkart
                            // Our clickable cells can be any of the count cells (1..3). Check all.
                            const clickableCells = [cells[1], cells[2], cells[3]].filter(Boolean);
                            const cellWithHandler = clickableCells.find(c => c.getAttribute && c.getAttribute('onclick'));
                            if (cellWithHandler) {
                                const onclickValue = cellWithHandler.getAttribute('onclick');
                                const extractedPersonnelId = parseInt(onclickValue.match(/\d+/)[0]);
                                if (extractedPersonnelId === personnelId) {
                                    targetRow = row;
                                }
                            }
                        }
                    });
                    
                    if (targetRow) {
                        // Kırılım satırları oluştur
                        details.forEach((detail, index) => {
                            const breakdownRow = document.createElement('tr');
                            breakdownRow.className = 'breakdown-row';
                            breakdownRow.style.backgroundColor = '#f8f9fa';
                            breakdownRow.style.borderLeft = '4px solid #007bff';
                            
                            breakdownRow.innerHTML = `
                                <td style="padding-left: 30px; font-style: italic; color: #666;">
                                    📅 ${detail.date}
                                </td>
                                <td colspan="2" style="color: #666;">
                                    ${detail.subject}
                                </td>
                                <td style="color: #666;">
                                    ${detail.count} adet
                                </td>
                                <td colspan="2" style="color: #666; font-size: 0.9em;">
                                    ${detail.topic}
                                </td>
                            `;
                            
                            // Hedef satırdan sonra ekle
                            targetRow.insertAdjacentElement('afterend', breakdownRow);
                            targetRow = breakdownRow; // Sonraki satır için referans güncelle
                        });
                        
                        console.log(`✅ ${details.length} breakdown satırı eklendi`);
                    }
                } else {
                    console.error('❌ Veri alınamadı!');
                }
            })
            .catch(error => {
                console.error('❌ Error fetching breakdown data:', error);
            });
    } catch (error) {
        console.error('❌ Error in showTrainingFeedbackBreakdown:', error);
    }
}

// Global fonksiyon olarak tanımla
// Not: editTrainingFeedbackRecord function is declared below in global scope;
// in non-module scripts it is already accessible as window.editTrainingFeedbackRecord.

function resetTrainingFeedbackModal() {
    const modal = document.getElementById('addTrainingFeedbackModal');
    if (!modal) return;
    
    // Modal başlığını normal haline döndür
    const modalTitle = modal.querySelector('.modal-header h3');
    if (modalTitle) {
    modalTitle.innerHTML = '<i class="fas fa-graduation-cap"></i> Yeni Eğitim Kaydı Ekle';
    }
    
    // Submit butonunu normal haline döndür
    const submitBtn = modal.querySelector('button[type="submit"]');
    if (submitBtn) {
        submitBtn.innerHTML = '<i class="fas fa-plus"></i> Kaydet';
    }
    
    // Form'u temizle
    const form = modal.querySelector('form');
    if (form) {
        form.reset();
        
        // Hidden editRecordId field'ını kaldır
        const editIdField = form.querySelector('input[name="editRecordId"]');
        if (editIdField) {
            editIdField.remove();
        }
    }
    
    console.log('✅ Modal reset edildi - normal ekleme modunda');
}

async function editTrainingFeedbackRecord(recordId) {
    console.log(`✏️ Editing training feedback record with ID: ${recordId}`);
    
    try {
        // Kayıt bilgilerini getir
        const response = await fetch(`${API_BASE_URL}/api/training-feedback/${recordId}`);
        if (response.ok) {
            const result = await response.json();
            const record = result.data;
            
            console.log('📋 Training feedback record data loaded:', record);
            
            // Formu doldur
            document.getElementById('editTrainingFeedbackId').value = record.id;
            document.getElementById('edit_training_feedback_date').value = record.date || '';
            document.getElementById('edit_training_feedback_personnel_id').value = record.personnel_id || '';
            // Yeni şema alanları
            document.getElementById('edit_feedback_count').value = record.feedback_count || '';
            document.getElementById('edit_feedback_subject').value = record.feedback_subject || '';
            document.getElementById('edit_general_training_count').value = record.general_training_count || '';
            document.getElementById('edit_general_training_subject').value = record.general_training_subject || '';
            // Kayıtlarda birebir eğitim alanları 'personal_*' olarak tutuluyor
            document.getElementById('edit_one_on_one_training_count').value = record.personal_training_count || '';
            document.getElementById('edit_one_on_one_training_subject').value = record.personal_training_subject || '';
            
            // Personnel dropdown'unu doldur
            await loadPersonnelOptions('edit_training_feedback_personnel_id');
            document.getElementById('edit_training_feedback_personnel_id').value = record.personnel_id || '';
            
            // Modal'ı aç
            openModal('editTrainingFeedbackModal');
        } else {
            throw new Error('Training feedback record not found');
        }
    } catch (error) {
        console.error('❌ Error loading training feedback record for edit:', error);
        alert('Kayıt bilgileri yüklenirken hata oluştu: ' + error.message);
    }
}

function closeAllModals() {
    // Tek kaynak: mevcut kapatma yardımcılarını kullan
    try { closeModals(); } catch (e) {
        // Fallback (eski yöntemler)
        try {
            document.querySelectorAll('.modal-overlay').forEach(overlay => { overlay.classList.remove('active'); overlay.style.display = 'none'; });
            document.querySelectorAll('[id$="Modal"]').forEach(modal => { modal.style.display = 'none'; });
            document.querySelectorAll('[id$="backdrop"]').forEach(backdrop => { backdrop.style.display = 'none'; });
            document.body.classList.remove('modal-open');
            document.body.style.overflow = '';
        } catch {}
    }
}

async function fillEditForm(recordId) {
    console.log('🔧 fillEditForm başladı, recordId:', recordId);
    
    try {
        // API'dan veriyi al
        console.log('🌐 API çağrısı yapılıyor...');
        const response = await fetch(`${API_BASE_URL}/api/training-feedback`);
        const result = await response.json();
        
        if (!response.ok) {
            console.error('API hatası:', result);
            alert('API hatası: ' + result.message);
            return;
        }
        
        console.log('✅ API yanıtı alındı:', result);
        
        // Kaydı bul
        const record = result.data.find(r => r.id === recordId);
        if (!record) {
            alert('Kayıt bulunamadı! ID: ' + recordId);
            return;
        }
        
        console.log('✅ Edit edilecek kayıt:', record);
        
        // Form elemanlarını doldur
        const personnelSelect = document.getElementById('training_feedback_personnel_id');
        const dateInput = document.getElementById('training_feedback_date');
    // Legacy fields removed in new schema (uyarı/kesinti yok)
    const warningTypeSelect = null;
    const warningSubjectInput = null;
        
        console.log('📝 Form elemanları:', {
            personnelSelect: !!personnelSelect,
            dateInput: !!dateInput,
            warningTypeSelect: !!warningTypeSelect,
            warningSubjectInput: !!warningSubjectInput
        });
        
        if (personnelSelect) personnelSelect.value = record.personnel_id || '';
        if (dateInput) dateInput.value = record.date || '';
    // No warning fields to populate in new schema
        
        // Diğer alanları doldur
        const feedbackCount = document.getElementById('feedback_count');
        const feedbackSubject = document.getElementById('feedback_subject');
        const generalTrainingCount = document.getElementById('general_training_count');
        const generalTrainingSubject = document.getElementById('general_training_subject');
        const oneOnOneCount = document.getElementById('one_on_one_training_count');
        const oneOnOneSubject = document.getElementById('one_on_one_training_subject');
        
        if (feedbackCount) feedbackCount.value = record.feedback_count || 0;
        if (feedbackSubject) feedbackSubject.value = record.feedback_subject || '';
        if (generalTrainingCount) generalTrainingCount.value = record.general_training_count || 0;
        if (generalTrainingSubject) generalTrainingSubject.value = record.general_training_subject || '';
        if (oneOnOneCount) oneOnOneCount.value = record.personal_training_count || 0;
        if (oneOnOneSubject) oneOnOneSubject.value = record.personal_training_subject || '';
        
        // Submit butonunu güncelleme moduna getir
        const modal = document.getElementById('addTrainingFeedbackModal');
        const submitBtn = modal.querySelector('.btn-success');
        if (submitBtn) {
            submitBtn.textContent = 'Güncelle';
            // Doğrudan onclick yerine güncelleme fonksiyonuna yönlendir
            submitBtn.onclick = () => updateTrainingFeedbackRecord(recordId);
        }
        
        console.log('Form dolduruldu');
        
    } catch (error) {
        console.error('Form doldurma hatası:', error);
        alert('Form doldurma hatası: ' + error.message);
    }
}

async function updateTrainingFeedbackRecord(recordId) {
    console.log(`🔄 Updating training-feedback record: ${recordId}`);
    
    try {
    const formData = {
            personnel_id: parseInt(document.getElementById('training_feedback_personnel_id').value),
            date: document.getElementById('training_feedback_date').value,
            feedback_count: parseInt(document.getElementById('feedback_count').value) || 0,
            feedback_subject: document.getElementById('feedback_subject').value,
            general_training_count: parseInt(document.getElementById('general_training_count').value) || 0,
            general_training_subject: document.getElementById('general_training_subject').value,
            personal_training_count: parseInt(document.getElementById('one_on_one_training_count').value) || 0,
            personal_training_subject: document.getElementById('one_on_one_training_subject').value,
            notes: "" // HTML'de notes field'ı yok, boş bırak
        };
        
        console.log('📤 Updating record with data:', formData);
        
        const response = await fetch(`${API_BASE_URL}/api/training-feedback/${recordId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            console.log('✅ Training-feedback record updated successfully:', result);
        showNotification('✅ Eğitim kaydı başarıyla güncellendi!', 'success');
            
            // Modal'ı kapat ve formu temizle
            closeTrainingFeedbackModal();
            
            // Sayfayı yenile
            await updateTrainingFeedbackContent();
        } else {
            console.error('❌ Update failed:', result);
            alert(`❌ Güncelleme başarısız: ${result.detail || 'Bilinmeyen hata'}`);
        }
        
    } catch (error) {
        console.error('❌ Error updating training-feedback record:', error);
        alert('❌ Güncelleme sırasında hata oluştu! Backend sunucusu çalışıyor mu?');
    }
}

function closeTrainingFeedbackModal() {
    console.log('🔒 Closing training feedback modal');
    
    const modal = document.getElementById('addTrainingFeedbackModal');
    if (!modal) return;
    
    const overlay = modal.closest('.modal-overlay');
    
    if (overlay) {
        overlay.classList.remove('active');
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 400);
    } else {
        modal.style.display = 'none';
    }
    
    // Form alanlarını temizle
    try {
        const resetIds = [
            'training_feedback_personnel_id',
            'training_feedback_date',
            'feedback_count',
            'feedback_subject',
            'general_training_count',
            'general_training_subject',
            'one_on_one_training_count',
            'one_on_one_training_subject'
        ];
        resetIds.forEach(id => { const el = document.getElementById(id); if (el) el.value = el.type === 'number' ? 0 : ''; });
        // Submit butonunu sıfırla
        const submitBtn = modal.querySelector('.btn-success');
        if (submitBtn) {
            submitBtn.textContent = 'Kaydet';
            // Varsayılan form submit akışına dön (global submit dinleyicisi handleFormSubmitInline çağıracak)
            submitBtn.onclick = null;
        }
    } catch (error) {
        console.log('Form temizleme hatası:', error);
    }
}

// Ekleme işlemi için wrapper fonksiyon
function addTrainingFeedbackRecord() {
    const form = document.getElementById('addTrainingFeedbackForm');
    if (form) {
    handleFormSubmitInline(form);
    }
}

async function deleteTrainingFeedbackRecord(recordId) {
    if (!confirm('Bu eğitim kaydını silmek istediğinizden emin misiniz?')) {
        return;
    }
    
    try {
        console.log(`🗑️ Deleting training-feedback record: ${recordId}`);
        // Optimistic UI removal
        const row = document.querySelector(`tr[data-record-id="${recordId}"]`);
        if (row && row.parentElement) {
            row.classList.add('fade-out');
            setTimeout(() => row.remove(), 200);
        }
        
        const response = await fetch(`${API_BASE_URL}/api/training-feedback/${recordId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            console.log('✅ Training-feedback record deleted successfully');
            showNotification('✅ Eğitim kaydı başarıyla silindi!', 'success');
            // Quick refresh to sync both detail and summary
            setTimeout(() => { updateTrainingFeedbackContent().catch(()=>{}); }, 150);
        } else {
            const errorResult = await response.json();
            console.error('❌ Delete failed:', errorResult);
            alert(`❌ Silme başarısız: ${errorResult.detail || 'Bilinmeyen hata'}`);
            await updateTrainingFeedbackContent();
        }
        
    } catch (error) {
        console.error('❌ Error deleting training-feedback record:', error);
        alert('❌ Silme sırasında hata oluştu! Backend sunucusu çalışıyor mu?');
        await updateTrainingFeedbackContent();
    }
}

// === Eğitim-Geribildirim Filtre Fonksiyonları ===
function getTrainingFeedbackFilters() {
    const startDate = document.getElementById('trainingFeedbackStartDate')?.value || '';
    const endDate = document.getElementById('trainingFeedbackEndDate')?.value || '';
    if (startDate && endDate && startDate > endDate) {
        alert('⚠️ Başlangıç tarihi bitiş tarihinden büyük olamaz!');
        return null;
    }
    return { startDate, endDate };
}

async function applyTrainingFeedbackFilter() {
    const filters = getTrainingFeedbackFilters();
    if (!filters) return;
    await updateTrainingFeedbackContent();
    showNotification('📅 Filtre uygulandı', 'success');
}

async function clearTrainingFeedbackFilter() {
    const ids = ['trainingFeedbackStartDate', 'trainingFeedbackEndDate'];
    ids.forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
    await updateTrainingFeedbackContent();
    showNotification('🧹 Filtreler temizlendi', 'info');
}

// Eksik personel fonksiyonları - butonlardan çağrılan wrapper fonksiyonlar
async function editPersonnelRecord(personnelId) {
    return await editPersonnel(personnelId);
}

async function deletePersonnelRecord(personnelId) {
    return await deletePersonnel(personnelId);
}

// Eksik günlük kayıt düzenleme fonksiyonu
async function editDailyRecord(recordId) {
    console.log(`✏️ Editing daily record with ID: ${recordId}`);
    
    try {
        // Kayıt bilgilerini getir
        const response = await fetch(`${API_BASE_URL}/api/daily-records/${recordId}`);
        if (response.ok) {
            const result = await response.json();
            const record = result.data;
            
            console.log('📋 Daily record data loaded:', record);
            
            // Form elemanları mevcut mu kontrol et
            const editModal = document.getElementById('editDailyRecordModal');
            if (!editModal) {
                alert('⚠️ Düzenleme modalı bulunamadı. Lütfen sayfayı yenileyin.');
                return;
            }

            // Formu doldur
            document.getElementById('editDailyRecordId').value = record.id;
            document.getElementById('editDailyRecordDate').value = record.date || '';

            await loadPersonnelOptions('editDailyRecordPersonnel');
            document.getElementById('editDailyRecordPersonnel').value = record.personnel_id;
            document.getElementById('editDailyRecordCallNumber').value = record.call_number || '';
            document.getElementById('editDailyRecordScore').value = record.score || '';
            document.getElementById('editDailyRecordNotes').value = record.notes || '';

            // Modal'ı aç
            openModal('editDailyRecordModal');
        } else {
            throw new Error('Kayıt bulunamadı');
        }
    } catch (error) {
        console.error('❌ Error loading daily record for edit:', error);
        alert('❌ Günlük kayıt bilgileri yüklenemedi!');
    }
}

// Eksik günlük kayıt silme fonksiyonu
async function deleteDailyRecord(recordId) {
    if (!confirm('Bu günlük kaydı silmek istediğinizden emin misiniz?')) {
        return;
    }
    
    try {
        console.log(`🗑️ Deleting daily record: ${recordId}`);
        // Optimistic remove in UI
        const row = document.querySelector(`tr[data-record-id="${recordId}"]`);
        if (row && row.parentElement) {
            row.classList.add('fade-out');
            setTimeout(() => row.remove(), 200);
        }
        
        const response = await fetch(`${API_BASE_URL}/api/daily-records/${recordId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            console.log('✅ Daily record deleted successfully');
            showNotification('✅ Günlük kayıt başarıyla silindi!', 'success');
            // Quick rebuild to sync summaries
            setTimeout(() => { updateRecordsContent().catch(()=>{}); }, 150);
        } else {
            const errorResult = await response.json();
            console.error('❌ Delete failed:', errorResult);
            alert(`❌ Silme başarısız: ${errorResult.detail || 'Bilinmeyen hata'}`);
            await updateRecordsContent();
        }
        
    } catch (error) {
        console.error('❌ Error deleting daily record:', error);
        showErrorToast('Silme sırasında hata oluştu! Backend sunucusu çalışıyor mu?');
        await updateRecordsContent();
    }
}

console.log('📄 app.js loaded completely - VERSION 6.30-FIXED-FUNCTIONS!');
