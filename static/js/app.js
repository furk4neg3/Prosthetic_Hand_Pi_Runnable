/* ═══════════════════════════════════════════════════════════════════════
   PROSTHETIC HAND CONTROL — CLIENT APP
   ═══════════════════════════════════════════════════════════════════════ */

// ─── State ──────────────────────────────────────────────────────────
let movements = [];
let currentFilter = 'all';
let systemReady = false;
let socket = null;
let toastTimeout = null;

// ─── DOM References ─────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── Initialize ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    loadMovements();
    checkStatus();
    initFilterTabs();
});

// ═══════════════════════════════════════════════════════════════════════
// SOCKET.IO CONNECTION
// ═══════════════════════════════════════════════════════════════════════
function initSocket() {
    try {
        socket = io();

        socket.on('connect', () => {
            console.log('🔌 Socket connected');
            // Re-check system status on (re)connect
            checkStatus();
        });

        socket.on('disconnect', () => {
            console.warn('🔌 Socket disconnected');
            systemReady = false;
            const badge = $('#system-status-badge');
            badge.className = 'status-badge disconnected';
            badge.querySelector('.status-text').textContent = 'Disconnected';
            $('#servo-mode-badge').classList.add('stale');
        });

        socket.on('connect_error', () => {
            console.warn('🔌 Socket connection error');
            systemReady = false;
            const badge = $('#system-status-badge');
            badge.className = 'status-badge disconnected';
            badge.querySelector('.status-text').textContent = 'Disconnected';
            $('#servo-mode-badge').classList.add('stale');
        });

        socket.on('status_update', (data) => {
            updateToast(data.state, data.message);
            // Only update pipeline for the slow phases (moving, holding, resting)
            // Fast phases are animated client-side
            if (['moving', 'holding', 'resting', 'ready', 'error'].includes(data.state)) {
                updatePipelineLive(data.state, data.message);
            }
        });

        socket.on('inference_result', (data) => {
            showResults(data);
        });

        socket.on('servo_update', (data) => {
            updateServoGauges(data.angles);
        });
    } catch (e) {
        console.warn('Socket.IO not available, using polling');
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════════════════
async function loadMovements() {
    try {
        const res = await fetch('/api/movements');
        movements = await res.json();
        renderMovementCards();
    } catch (e) {
        console.error('Failed to load movements:', e);
    }
}

async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const status = await res.json();

        const badge = $('#system-status-badge');
        const modeBadge = $('#servo-mode-badge');

        if (status.initialized) {
            badge.className = 'status-badge ready';
            badge.querySelector('.status-text').textContent = 'System Ready';
            systemReady = true;
        } else {
            badge.className = 'status-badge error';
            badge.querySelector('.status-text').textContent = 'Not Initialized';
        }

        if (status.servo_mode === 'hardware') {
            modeBadge.className = 'servo-mode-badge hardware';
            modeBadge.querySelector('span').textContent = 'HW';
        } else {
            modeBadge.className = 'servo-mode-badge';
            modeBadge.querySelector('span').textContent = 'SIM';
        }
        // Clear stale state — we just got a fresh response from the server
        modeBadge.classList.remove('stale');
    } catch (e) {
        const badge = $('#system-status-badge');
        badge.className = 'status-badge error';
        badge.querySelector('.status-text').textContent = 'Connection Error';
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RENDER MOVEMENT CARDS
// ═══════════════════════════════════════════════════════════════════════
function renderMovementCards() {
    const grid = $('#movements-grid');
    grid.innerHTML = '';

    const exerciseNames = { 1: 'Finger', 2: 'Hand', 3: 'Grasp' };

    movements.forEach((mov, i) => {
        // Filter check
        if (currentFilter !== 'all' && mov.exercise !== parseInt(currentFilter)) {
            return;
        }

        const card = document.createElement('div');
        card.className = 'movement-card';
        card.style.animationDelay = `${i * 0.03}s`;
        card.dataset.index = mov.encoded;
        card.id = `movement-card-${mov.encoded}`;

        const exClass = `ex${mov.exercise}`;
        const exName = exerciseNames[mov.exercise] || `E${mov.exercise}`;

        let imageContent;
        if (mov.image) {
            imageContent = `<img src="${mov.image}" alt="${mov.name}" loading="lazy">`;
        } else {
            imageContent = `
                <div class="card-image-placeholder">
                    <span class="placeholder-icon">✋</span>
                    <span>Label ${mov.original}</span>
                </div>`;
        }

        card.innerHTML = `
            <div class="card-number">${mov.encoded + 1}</div>
            <div class="card-exercise ${exClass}">${exName}</div>
            <div class="card-image">${imageContent}</div>
            <div class="card-body">
                <div class="card-title">${mov.name}</div>
                <div class="card-meta">
                    <span class="card-label">Label ${mov.original}</span>
                    <span class="card-action">Execute →</span>
                </div>
            </div>
        `;

        card.addEventListener('click', () => executeMovement(mov.encoded));
        grid.appendChild(card);
    });
}

// ═══════════════════════════════════════════════════════════════════════
// FILTER TABS
// ═══════════════════════════════════════════════════════════════════════
function initFilterTabs() {
    $$('.filter-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.filter-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            currentFilter = tab.dataset.filter;

            // Update description
            $$('.filter-desc').forEach(d => d.classList.remove('active'));
            const desc = $(`.filter-desc[data-for="${currentFilter}"]`);
            if (desc) desc.classList.add('active');

            renderMovementCards();
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════
// EXECUTE MOVEMENT — non-blocking, cards stay interactive
// ═══════════════════════════════════════════════════════════════════════
async function executeMovement(index) {
    if (!systemReady) {
        alert('System is not ready yet. Please wait for initialization.');
        return;
    }

    const movName = movements[index].name;

    // Show pipeline with smooth client-side animation for the fast steps
    showPipelineSweep(movName);

    // Show small toast
    showToast(movName, 'Processing sEMG data...');

    // Update header badge to busy
    const badge = $('#system-status-badge');
    badge.className = 'status-badge busy';
    badge.querySelector('.status-text').textContent = 'Executing';

    try {
        const res = await fetch(`/api/execute/${index}`, { method: 'POST' });
        const data = await res.json();

        if (data.error) {
            showToast('Error', data.error);
            hideToastAfter(3000);
            badge.className = 'status-badge ready';
            badge.querySelector('.status-text').textContent = 'System Ready';
        }
    } catch (e) {
        console.error('Execution failed:', e);
        showToast('Error', 'Failed to execute movement');
        hideToastAfter(3000);
        badge.className = 'status-badge ready';
        badge.querySelector('.status-text').textContent = 'System Ready';
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATION — small, bottom-right, non-blocking
// ═══════════════════════════════════════════════════════════════════════
function showToast(title, message) {
    const toast = $('#execution-toast');
    // Cancel any pending hide
    if (toastTimeout) {
        clearTimeout(toastTimeout);
        toastTimeout = null;
    }
    toast.classList.remove('hidden', 'fade-out', 'done');
    $('#toast-title').textContent = title;
    $('#toast-message').textContent = message;
}

function hideToastAfter(ms) {
    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        const toast = $('#execution-toast');
        toast.classList.add('fade-out');
        setTimeout(() => toast.classList.add('hidden'), 300);
    }, ms);
}

function updateToast(state, message) {
    const toast = $('#execution-toast');
    if (toast.classList.contains('hidden')) return;

    $('#toast-message').textContent = message;

    if (state === 'ready') {
        toast.classList.add('done');
        $('#toast-title').textContent = '✅ Complete';
        hideToastAfter(2000);

        // Restore header badge
        const badge = $('#system-status-badge');
        badge.className = 'status-badge ready';
        badge.querySelector('.status-text').textContent = 'System Ready';
    } else if (state === 'error') {
        $('#toast-title').textContent = '❌ Error';
        hideToastAfter(4000);

        const badge = $('#system-status-badge');
        badge.className = 'status-badge error';
        badge.querySelector('.status-text').textContent = 'Error';
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PIPELINE VISUALIZATION — smooth animated sweep
// ═══════════════════════════════════════════════════════════════════════
const PIPELINE_ORDER = ['sampling', 'preprocessing', 'inferring', 'moving', 'holding', 'resting'];

/**
 * Animate the first 3 pipeline steps (sample → preprocess → inference)
 * with staggered delays on the client side, since they complete in ~6ms
 * total on the server. This makes the pipeline feel alive and intentional.
 * Steps 4-6 (moving, holding, resting) are updated live from server events.
 */
function showPipelineSweep(movementName) {
    const pipelineSection = $('#pipeline-section');
    pipelineSection.classList.remove('hidden');

    const steps = $$('.pipeline-step');
    const connectors = $$('.pipeline-connector');

    // Reset all steps
    steps.forEach(s => s.classList.remove('active', 'done', 'sweep', 'sweep-done'));
    connectors.forEach(c => c.classList.remove('done', 'sweep-done'));
    $('#pipeline-message').textContent = `Processing: ${movementName}`;

    // Animate first 3 steps with staggered timing (150ms each)
    const fastStepDelay = 150; // ms per step

    for (let i = 0; i < 3; i++) {
        // Highlight this step
        setTimeout(() => {
            // Mark previous step as done
            if (i > 0) {
                steps[i - 1].classList.remove('sweep');
                steps[i - 1].classList.add('sweep-done');
                if (connectors[i - 1]) connectors[i - 1].classList.add('sweep-done');
            }
            // Highlight current step
            steps[i].classList.add('sweep');

            const labels = ['Getting sEMG sample...', 'Preprocessing data...', 'Running inference...'];
            $('#pipeline-message').textContent = labels[i];
        }, i * fastStepDelay);
    }

    // After the 3 fast steps are done, mark inference as done too
    setTimeout(() => {
        steps[2].classList.remove('sweep');
        steps[2].classList.add('sweep-done');
        if (connectors[2]) connectors[2].classList.add('sweep-done');
    }, 3 * fastStepDelay);
}

/**
 * Update pipeline for the live (slow) phases from server events.
 * Called for: moving, holding, resting, ready
 */
function updatePipelineLive(state, message) {
    const steps = $$('.pipeline-step');
    const connectors = $$('.pipeline-connector');
    const stateIndex = PIPELINE_ORDER.indexOf(state);

    if (stateIndex < 3) return; // Fast steps are handled by sweep animation

    // Ensure fast steps are all marked done
    for (let i = 0; i < 3; i++) {
        steps[i].classList.remove('active', 'sweep');
        steps[i].classList.add('sweep-done');
        if (connectors[i]) connectors[i].classList.add('sweep-done');
    }

    // Update slow steps
    for (let i = 3; i < steps.length; i++) {
        steps[i].classList.remove('active', 'done', 'sweep', 'sweep-done');
        if (i < stateIndex) {
            steps[i].classList.add('sweep-done');
            if (connectors[i]) connectors[i].classList.add('sweep-done');
        } else if (i === stateIndex) {
            steps[i].classList.add('sweep');
        }
    }

    // Update connectors for slow steps
    for (let i = 3; i < connectors.length; i++) {
        if (i < stateIndex) {
            connectors[i].classList.add('sweep-done');
        }
    }

    $('#pipeline-message').textContent = message;

    // If ready, mark all done briefly then fade
    if (state === 'ready') {
        steps.forEach(s => {
            s.classList.remove('sweep');
            s.classList.add('sweep-done');
        });
        connectors.forEach(c => c.classList.add('sweep-done'));
        $('#pipeline-message').textContent = 'Complete — ready for next movement';
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RESULTS DISPLAY
// ═══════════════════════════════════════════════════════════════════════
function showResults(data) {
    const resultsSection = $('#results-section');
    resultsSection.classList.remove('hidden');

    // Inference results
    $('#result-selected').textContent = data.selected.name;
    $('#result-predicted').textContent = data.prediction.name;

    const conf = (data.prediction.confidence * 100).toFixed(1);
    const confBar = $('#confidence-bar');
    confBar.style.setProperty('--confidence', `${conf}%`);
    $('#confidence-value').textContent = `${conf}%`;

    $('#result-time').textContent = `${data.prediction.inference_time_ms.toFixed(1)} ms`;

    const matchEl = $('#result-match');
    if (data.prediction.correct) {
        matchEl.textContent = '✅ Correct';
        matchEl.className = 'result-value correct';
    } else {
        matchEl.textContent = '❌ Mismatch';
        matchEl.className = 'result-value incorrect';
    }

    // Sample badge
    $('#sample-badge').textContent = `Sample ${data.sample_index}/${data.total_samples}`;

    // Draw EMG
    drawEMG(data.emg_data);

    // Update servo gauges from the predicted movement
    const predictedMovement = movements[data.prediction.index];
    if (predictedMovement) {
        updateServoGauges(predictedMovement.angles);
    }

    // Update toast with inference info
    const confStr = (data.prediction.confidence * 100).toFixed(0);
    showToast(
        data.prediction.name,
        `Confidence: ${confStr}% · ${data.prediction.inference_time_ms.toFixed(1)}ms`
    );
}

// ═══════════════════════════════════════════════════════════════════════
// SERVO GAUGES
// ═══════════════════════════════════════════════════════════════════════
function updateServoGauges(angles) {
    const fingers = ['thumb', 'index', 'middle', 'ring', 'little'];
    fingers.forEach(name => {
        const gauge = $(`#gauge-${name}`);
        if (!gauge) return;
        const angle = angles[name] || 0;
        const pct = (angle / 180) * 100;
        gauge.querySelector('.gauge-fill').style.setProperty('--fill', `${pct}%`);
        gauge.querySelector('.gauge-value').textContent = `${angle}°`;
    });
}

// ═══════════════════════════════════════════════════════════════════════
// EMG VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════
function drawEMG(emgData) {
    const canvas = $('#emg-canvas');
    if (!canvas || !emgData || emgData.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;

    ctx.clearRect(0, 0, W, H);

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let y = 0; y < H; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }
    for (let x = 0; x < W; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
    }

    const nSamples = emgData.length;
    const nChannels = emgData[0].length;
    const channelsToShow = Math.min(nChannels, 10);

    // Find global min/max for scaling
    let globalMin = Infinity, globalMax = -Infinity;
    for (let t = 0; t < nSamples; t++) {
        for (let ch = 0; ch < channelsToShow; ch++) {
            const v = emgData[t][ch];
            if (v < globalMin) globalMin = v;
            if (v > globalMax) globalMax = v;
        }
    }
    const range = globalMax - globalMin || 1;

    // Color palette for channels
    const colors = [
        '#6C5CE7', '#00CEC9', '#FD79A8', '#FDCB6E', '#00B894',
        '#E17055', '#0984E3', '#A29BFE', '#81ECEC', '#FAB1A0'
    ];

    // Draw each channel
    for (let ch = 0; ch < channelsToShow; ch++) {
        ctx.beginPath();
        ctx.strokeStyle = colors[ch % colors.length];
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.7;

        for (let t = 0; t < nSamples; t++) {
            const x = (t / (nSamples - 1)) * W;
            const normalized = (emgData[t][ch] - globalMin) / range;
            const y = H - (normalized * H * 0.85 + H * 0.075);

            if (t === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    ctx.globalAlpha = 1.0;

    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(`${nSamples} samples × ${channelsToShow} channels`, 8, H - 6);
}
