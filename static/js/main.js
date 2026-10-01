/* Main Client-Side JavaScript Logic */

document.addEventListener('DOMContentLoaded', () => {
    // Mobile navigation toggle
    const menuBtn = document.getElementById('mobile-menu-btn');
    const mobileMenu = document.getElementById('mobile-menu');
    if (menuBtn && mobileMenu) {
        menuBtn.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }
});

// --- DATASET PAGE LOGIC ---
async function fetchDatasetInfo() {
    try {
        const response = await fetch('/dataset-info');
        const data = await response.json();
        if (!data.success) return;

        // Populate summary cards
        document.getElementById('summary-rows').textContent = data.shape[0];
        document.getElementById('summary-cols').textContent = data.shape[1];
        document.getElementById('summary-missing').textContent = data.total_missing;
        document.getElementById('summary-duplicates').textContent = data.duplicates_count;
        document.getElementById('active-filename').textContent = `File: ${data.file_name}`;

        // Preview Table Headers
        const theadTr = document.getElementById('preview-thead-tr');
        if (theadTr) {
            theadTr.innerHTML = data.columns.map(col => `<th class="px-4 py-3 font-semibold">${col}</th>`).join('');
        }

        // Preview Table Rows
        const tbody = document.getElementById('preview-tbody');
        if (tbody) {
            tbody.innerHTML = data.preview.map(row => {
                const cells = data.columns.map(col => `<td class="px-4 py-3 font-mono">${row[col] !== null ? row[col] : '<span class="text-amber-400">NaN</span>'}</td>`).join('');
                return `<tr class="hover:bg-slate-800/40 transition-colors">${cells}</tr>`;
            }).join('');
        }

        // Data Types Table
        const dtypesTbody = document.getElementById('dtypes-tbody');
        if (dtypesTbody) {
            dtypesTbody.innerHTML = data.columns.map(col => {
                const dtype = data.data_types[col] || 'unknown';
                const missing = data.missing_values[col] || 0;
                return `<tr class="hover:bg-slate-800/40">
                    <td class="px-4 py-2 font-medium text-slate-200">${col}</td>
                    <td class="px-4 py-2 font-mono text-purple-400">${dtype}</td>
                    <td class="px-4 py-2 text-right font-mono ${missing > 0 ? 'text-amber-400 font-bold' : 'text-slate-400'}">${missing}</td>
                </tr>`;
            }).join('');
        }

        // Statistics Table
        const statsTbody = document.getElementById('stats-tbody');
        if (statsTbody && data.summary_statistics) {
            const stats = data.summary_statistics;
            const features = Object.keys(stats);
            statsTbody.innerHTML = features.map(feat => {
                const fStats = stats[feat];
                return `<tr class="hover:bg-slate-800/40">
                    <td class="px-3 py-2 font-medium text-slate-200">${feat}</td>
                    <td class="px-3 py-2 text-right font-mono text-slate-300">${fStats.mean || '-'}</td>
                    <td class="px-3 py-2 text-right font-mono text-slate-300">${fStats.std || '-'}</td>
                    <td class="px-3 py-2 text-right font-mono text-slate-300">${fStats.min || '-'}</td>
                    <td class="px-3 py-2 text-right font-mono text-slate-300">${fStats.max || '-'}</td>
                </tr>`;
            }).join('');
        }

    } catch (err) {
        console.error('Error fetching dataset info:', err);
    }
}

// --- PREPROCESSING PAGE LOGIC ---
async function triggerPreprocessing() {
    try {
        const response = await fetch('/preprocess', { method: 'POST' });
        const data = await response.json();
        if (!data.success) return;

        const summary = data.summary;
        const missingElem = document.getElementById('prep-missing-count');
        const dupElem = document.getElementById('prep-dup-count');
        const trainElem = document.getElementById('split-train');
        const testElem = document.getElementById('split-test');
        const totalElem = document.getElementById('split-total');

        if (missingElem) missingElem.textContent = `${summary.missing_count_handled} values imputed`;
        if (dupElem) dupElem.textContent = `${summary.duplicates_removed} records purged`;
        if (totalElem) totalElem.textContent = summary.final_shape[0];
        if (trainElem) trainElem.textContent = Math.round(summary.final_shape[0] * 0.8);
        if (testElem) testElem.textContent = Math.round(summary.final_shape[0] * 0.2);

    } catch (err) {
        console.error('Error running preprocessing:', err);
    }
}

// --- TRAINING PAGE LOGIC ---
async function triggerTraining() {
    const btn = document.getElementById('btn-train-models');
    const statusText = document.getElementById('models-status-text');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner animate-spin mr-2"></i>Training Models...`;
    }
    if (statusText) statusText.textContent = "Training models in progress...";

    try {
        const response = await fetch('/train', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            if (statusText) statusText.textContent = "All 4 Models Trained & Saved Successfully!";
            loadTrainingStats();
        }
    } catch (err) {
        console.error('Error training models:', err);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-play text-xs mr-2"></i>Re-Train All Models`;
        }
    }
}

async function loadTrainingStats() {
    try {
        const response = await fetch('/evaluation');
        const data = await response.json();
        if (!data.success) return;

        const metrics = data.metrics;
        
        // Update quick accuracy badges
        if (metrics['Logistic Regression']) document.getElementById('acc-lr').textContent = `${metrics['Logistic Regression'].accuracy}%`;
        if (metrics['Decision Tree']) document.getElementById('acc-dt').textContent = `${metrics['Decision Tree'].accuracy}%`;
        if (metrics['Random Forest']) document.getElementById('acc-rf').textContent = `${metrics['Random Forest'].accuracy}%`;
        if (metrics['K-Nearest Neighbors']) document.getElementById('acc-knn').textContent = `${metrics['K-Nearest Neighbors'].accuracy}%`;

        // Populate table
        const tbody = document.getElementById('train-summary-tbody');
        if (tbody) {
            const rows = [
                { name: 'Logistic Regression', type: 'Multinomial Linear', file: 'logistic_regression.joblib' },
                { name: 'Decision Tree', type: 'Binary Decision Splits', file: 'decision_tree.joblib' },
                { name: 'Random Forest', type: 'Ensemble 100 Trees', file: 'random_forest.joblib' },
                { name: 'K-Nearest Neighbors', type: 'Instance Distance (k=5)', file: 'k-nearest_neighbors.joblib' }
            ];

            tbody.innerHTML = rows.map(r => {
                const m = metrics[r.name] || { accuracy: '-', f1_weighted: '-' };
                return `<tr class="hover:bg-slate-800/40">
                    <td class="px-4 py-3 font-semibold text-slate-100 font-sans">${r.name}</td>
                    <td class="px-4 py-3 text-slate-400 font-sans">${r.type}</td>
                    <td class="px-4 py-3 text-right font-bold text-blue-400">${m.accuracy}%</td>
                    <td class="px-4 py-3 text-right font-bold text-purple-400">${m.f1_weighted}%</td>
                    <td class="px-4 py-3 text-right text-emerald-400 text-xs">${r.file}</td>
                </tr>`;
            }).join('');
        }

    } catch (err) {
        console.error('Error loading training stats:', err);
    }
}

// --- EVALUATION PAGE LOGIC ---
async function loadEvaluationData() {
    try {
        const response = await fetch('/evaluation');
        const data = await response.json();
        if (!data.success) return;

        const metrics = data.metrics;

        // Find top model
        let topModel = '';
        let topAcc = 0;
        for (const [name, m] of Object.entries(metrics)) {
            if (m.accuracy > topAcc) {
                topAcc = m.accuracy;
                topModel = name;
            }
        }
        const topBadge = document.getElementById('top-model-name');
        if (topBadge) topBadge.textContent = `${topModel} (${topAcc}%)`;

        // Populate Metrics Table
        const tbody = document.getElementById('eval-metrics-tbody');
        if (tbody) {
            const sortedModels = Object.keys(metrics).sort((a, b) => metrics[b].accuracy - metrics[a].accuracy);
            tbody.innerHTML = sortedModels.map((name, idx) => {
                const m = metrics[name];
                const rankBadge = idx === 0 ? '<span class="text-amber-400 font-bold"><i class="fa-solid fa-crown mr-1"></i>Rank 1</span>' : `Rank ${idx + 1}`;
                return `<tr class="hover:bg-slate-800/40">
                    <td class="px-4 py-3 font-semibold text-slate-100 font-sans">${name}</td>
                    <td class="px-4 py-3 text-right font-bold text-emerald-400">${m.accuracy}%</td>
                    <td class="px-4 py-3 text-right font-bold text-blue-400">${m.precision_weighted}%</td>
                    <td class="px-4 py-3 text-right font-bold text-purple-400">${m.recall_weighted}%</td>
                    <td class="px-4 py-3 text-right font-bold text-amber-400">${m.f1_weighted}%</td>
                    <td class="px-4 py-3 text-right font-sans">${rankBadge}</td>
                </tr>`;
            }).join('');
        }

        // Render Confusion Matrices Grid
        const cmGrid = document.getElementById('confusion-matrices-grid');
        if (cmGrid) {
            const modelsList = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors'];
            cmGrid.innerHTML = modelsList.map(name => {
                const cm = metrics[name] ? metrics[name].confusion_matrix : [[0,0,0],[0,0,0],[0,0,0]];
                return `
                <div class="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-3">
                    <div class="flex items-center justify-between">
                        <span class="font-heading font-bold text-slate-100 text-sm">${name}</span>
                        <span class="text-xs text-emerald-400 font-mono font-bold">${metrics[name] ? metrics[name].accuracy : 0}% Acc</span>
                    </div>

                    <!-- 3x3 Grid Matrix -->
                    <div class="grid grid-cols-4 gap-1 text-center text-xs font-mono">
                        <div class="text-[10px] text-slate-500 flex items-center justify-center">Act \\ Pred</div>
                        <div class="bg-slate-950 p-1.5 rounded text-red-400 font-bold">Poor</div>
                        <div class="bg-slate-950 p-1.5 rounded text-amber-400 font-bold">Avg</div>
                        <div class="bg-slate-950 p-1.5 rounded text-emerald-400 font-bold">Good</div>

                        <div class="bg-slate-950 p-1.5 rounded text-red-400 font-bold flex items-center justify-center">Poor</div>
                        <div class="p-2 rounded font-bold ${cm[0][0] > 0 ? 'bg-emerald-500/30 text-emerald-300 ring-1 ring-emerald-500/50' : 'bg-slate-950 text-slate-500'}">${cm[0][0]}</div>
                        <div class="p-2 rounded font-bold ${cm[0][1] > 0 ? 'bg-amber-500/20 text-amber-300' : 'bg-slate-950 text-slate-500'}">${cm[0][1]}</div>
                        <div class="p-2 rounded font-bold ${cm[0][2] > 0 ? 'bg-red-500/20 text-red-300' : 'bg-slate-950 text-slate-500'}">${cm[0][2]}</div>

                        <div class="bg-slate-950 p-1.5 rounded text-amber-400 font-bold flex items-center justify-center">Avg</div>
                        <div class="p-2 rounded font-bold ${cm[1][0] > 0 ? 'bg-red-500/20 text-red-300' : 'bg-slate-950 text-slate-500'}">${cm[1][0]}</div>
                        <div class="p-2 rounded font-bold ${cm[1][1] > 0 ? 'bg-emerald-500/30 text-emerald-300 ring-1 ring-emerald-500/50' : 'bg-slate-950 text-slate-500'}">${cm[1][1]}</div>
                        <div class="p-2 rounded font-bold ${cm[1][2] > 0 ? 'bg-amber-500/20 text-amber-300' : 'bg-slate-950 text-slate-500'}">${cm[1][2]}</div>

                        <div class="bg-slate-950 p-1.5 rounded text-emerald-400 font-bold flex items-center justify-center">Good</div>
                        <div class="p-2 rounded font-bold ${cm[2][0] > 0 ? 'bg-red-500/20 text-red-300' : 'bg-slate-950 text-slate-500'}">${cm[2][0]}</div>
                        <div class="p-2 rounded font-bold ${cm[2][1] > 0 ? 'bg-amber-500/20 text-amber-300' : 'bg-slate-950 text-slate-500'}">${cm[2][1]}</div>
                        <div class="p-2 rounded font-bold ${cm[2][2] > 0 ? 'bg-emerald-500/30 text-emerald-300 ring-1 ring-emerald-500/50' : 'bg-slate-950 text-slate-500'}">${cm[2][2]}</div>
                    </div>
                </div>`;
            }).join('');
        }

        // Render Charts via charts.js
        if (window.renderEvaluationCharts) {
            window.renderEvaluationCharts(metrics, data.feature_importances);
        }

    } catch (err) {
        console.error('Error loading evaluation data:', err);
    }
}

// --- PREDICTION PAGE LOGIC ---
function fillPreset(type) {
    if (type === 'top') {
        document.getElementById('Study_Hours').value = 8.5;
        document.getElementById('Attendance_Percentage').value = 95.0;
        document.getElementById('Previous_Semester_Marks').value = 88.0;
        document.getElementById('Assignment_Score').value = 92.0;
        document.getElementById('Internal_Assessment_Marks').value = 27.5;
        document.getElementById('Backlogs').value = 0;
        document.getElementById('Extracurricular_Activities').value = 'Yes';
        document.getElementById('Sleep_Hours').value = 8.0;
        document.getElementById('Internet_Usage_Hours').value = 2.0;
    } else if (type === 'average') {
        document.getElementById('Study_Hours').value = 5.0;
        document.getElementById('Attendance_Percentage').value = 75.0;
        document.getElementById('Previous_Semester_Marks').value = 65.0;
        document.getElementById('Assignment_Score').value = 70.0;
        document.getElementById('Internal_Assessment_Marks').value = 19.0;
        document.getElementById('Backlogs').value = 0;
        document.getElementById('Extracurricular_Activities').value = 'No';
        document.getElementById('Sleep_Hours').value = 6.5;
        document.getElementById('Internet_Usage_Hours').value = 3.5;
    } else if (type === 'atrisk') {
        document.getElementById('Study_Hours').value = 2.0;
        document.getElementById('Attendance_Percentage').value = 54.0;
        document.getElementById('Previous_Semester_Marks').value = 42.0;
        document.getElementById('Assignment_Score').value = 48.0;
        document.getElementById('Internal_Assessment_Marks').value = 12.0;
        document.getElementById('Backlogs').value = 2;
        document.getElementById('Extracurricular_Activities').value = 'No';
        document.getElementById('Sleep_Hours').value = 5.0;
        document.getElementById('Internet_Usage_Hours').value = 5.5;
    }
    document.getElementById('prediction-form').dispatchEvent(new Event('submit'));
}

async function handlePredictionSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const dataObj = {};
    formData.forEach((val, key) => dataObj[key] = val);

    const btn = document.getElementById('btn-predict');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner animate-spin mr-2"></i>Computing Prediction...`;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dataObj)
        });
        const res = await response.json();
        if (!res.success) return;

        // Model Used
        document.getElementById('output-model-used').textContent = res.selected_model;

        // Badge styling
        const badge = document.getElementById('output-category-badge');
        const category = res.predicted_category;
        badge.textContent = `${category} Performance`;
        
        if (category === 'Good') {
            badge.className = "inline-block px-8 py-3 rounded-2xl text-2xl font-extrabold font-heading bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 glow-badge-emerald";
        } else if (category === 'Average') {
            badge.className = "inline-block px-8 py-3 rounded-2xl text-2xl font-extrabold font-heading bg-amber-500/20 text-amber-400 border border-amber-500/40 glow-badge-purple";
        } else {
            badge.className = "inline-block px-8 py-3 rounded-2xl text-2xl font-extrabold font-heading bg-red-500/20 text-red-400 border border-red-500/40";
        }

        // Confidence Meter
        document.getElementById('output-confidence-val').textContent = `${res.confidence_percentage}%`;
        const bar = document.getElementById('output-confidence-bar');
        bar.style.width = `${res.confidence_percentage}%`;

        // Class Probabilities breakdown
        const probContainer = document.getElementById('probabilities-container');
        if (probContainer && res.probabilities) {
            probContainer.innerHTML = Object.entries(res.probabilities).map(([cls, pct]) => {
                return `<div class="flex items-center justify-between bg-slate-900 px-3 py-1.5 rounded border border-slate-800">
                    <span class="text-slate-300 font-sans">${cls}</span>
                    <span class="font-bold text-slate-100">${pct}%</span>
                </div>`;
            }).join('');
        }

        // Insights summary & points
        if (res.insights) {
            document.getElementById('insight-summary').textContent = res.insights.summary;
            const pointsContainer = document.getElementById('insight-key-points');
            if (pointsContainer && res.insights.key_points) {
                pointsContainer.innerHTML = res.insights.key_points.map(pt => {
                    const icon = pt.type === 'positive' ? 'fa-circle-check text-emerald-400' : (pt.type === 'negative' ? 'fa-circle-xmark text-red-400' : 'fa-triangle-exclamation text-amber-400');
                    return `<div class="flex items-start space-x-2 bg-slate-900/90 p-2.5 rounded-xl border border-slate-800 text-slate-300">
                        <i class="fa-solid ${icon} mt-0.5"></i>
                        <span>${pt.text}</span>
                    </div>`;
                }).join('');
            }
        }

    } catch (err) {
        console.error('Error submitting prediction:', err);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-calculator mr-2"></i>Predict Student Academic Performance`;
        }
    }
}

// --- HISTORY PAGE LOGIC ---
async function clearPredictionHistory() {
    if (!confirm('Are you sure you want to clear all prediction history records from SQLite database?')) return;
    try {
        const response = await fetch('/prediction-history', { method: 'DELETE' });
        const data = await response.json();
        if (data.success) {
            window.location.reload();
        }
    } catch (err) {
        console.error('Error clearing history:', err);
    }
}
