/* Chart.js Visualizations Handler */

let edaDataCache = null;
let charts = {};

async function loadEDAData() {
    try {
        const response = await fetch('/eda-data');
        const data = await response.json();
        if (!data.success) return;

        edaDataCache = data;

        renderTargetDistChart(data.target_distribution);
        renderAttendanceMarksChart(data.attendance_marks);
        renderStudyMarksChart(data.study_marks);
        renderHistogramChart();
        renderCorrelationMatrix(data.correlation_matrix);

    } catch (err) {
        console.error('Error loading EDA data:', err);
    }
}

function renderTargetDistChart(dist) {
    const ctx = document.getElementById('chart-target-dist');
    if (!ctx) return;

    if (charts.targetDist) charts.targetDist.destroy();

    const labels = Object.keys(dist);
    const counts = Object.values(dist);

    charts.targetDist = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                borderColor: '#0f172a',
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

function renderAttendanceMarksChart(dataPoints) {
    const ctx = document.getElementById('chart-attendance-marks');
    if (!ctx) return;

    if (charts.attendanceMarks) charts.attendanceMarks.destroy();

    const goodPts = dataPoints.filter(p => p.Performance_Category === 'Good').map(p => ({ x: p.Attendance_Percentage, y: p.Previous_Semester_Marks }));
    const avgPts = dataPoints.filter(p => p.Performance_Category === 'Average').map(p => ({ x: p.Attendance_Percentage, y: p.Previous_Semester_Marks }));
    const poorPts = dataPoints.filter(p => p.Performance_Category === 'Poor').map(p => ({ x: p.Attendance_Percentage, y: p.Previous_Semester_Marks }));

    charts.attendanceMarks = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Good', data: goodPts, backgroundColor: 'rgba(16, 185, 129, 0.7)' },
                { label: 'Average', data: avgPts, backgroundColor: 'rgba(245, 158, 11, 0.7)' },
                { label: 'Poor', data: poorPts, backgroundColor: 'rgba(239, 68, 68, 0.7)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Attendance Percentage (%)', color: '#94a3b8' },
                    grid: { color: 'rgba(51, 65, 85, 0.3)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    title: { display: true, text: 'Previous Semester Marks (%)', color: '#94a3b8' },
                    grid: { color: 'rgba(51, 65, 85, 0.3)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#94a3b8' } }
            }
        }
    });
}

function renderStudyMarksChart(dataPoints) {
    const ctx = document.getElementById('chart-study-marks');
    if (!ctx) return;

    if (charts.studyMarks) charts.studyMarks.destroy();

    // Group study hours into bins (1-3, 3-5, 5-7, 7-9, 9+)
    const bins = { '1-3 hrs': [], '3-5 hrs': [], '5-7 hrs': [], '7-9 hrs': [], '9+ hrs': [] };
    dataPoints.forEach(p => {
        const sh = p.Study_Hours;
        const marks = p.Previous_Semester_Marks;
        if (sh < 3) bins['1-3 hrs'].push(marks);
        else if (sh < 5) bins['3-5 hrs'].push(marks);
        else if (sh < 7) bins['5-7 hrs'].push(marks);
        else if (sh < 9) bins['7-9 hrs'].push(marks);
        else bins['9+ hrs'].push(marks);
    });

    const avgMarks = Object.keys(bins).map(key => {
        const arr = bins[key];
        return arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : 0;
    });

    charts.studyMarks = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(bins),
            datasets: [{
                label: 'Average Semester Marks (%)',
                data: avgMarks,
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: '#3b82f6',
                borderWidth: 1,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: 'rgba(51, 65, 85, 0.3)' }, ticks: { color: '#94a3b8' }, min: 40, max: 100 }
            },
            plugins: {
                legend: { labels: { color: '#94a3b8' } }
            }
        }
    });
}

function renderHistogramChart() {
    const ctx = document.getElementById('chart-feature-hist');
    const select = document.getElementById('hist-feature-select');
    if (!ctx || !select || !edaDataCache) return;

    if (charts.featureHist) charts.featureHist.destroy();

    const featName = select.value;
    const histInfo = edaDataCache.feature_distributions[featName];
    if (!histInfo) return;

    charts.featureHist = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: histInfo.bins.map(b => `${b}`),
            datasets: [{
                label: `Distribution Frequency: ${featName.replace('_', ' ')}`,
                data: histInfo.counts,
                backgroundColor: 'rgba(139, 92, 246, 0.7)',
                borderColor: '#8b5cf6',
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: 'rgba(51, 65, 85, 0.3)' }, ticks: { color: '#94a3b8' } }
            },
            plugins: {
                legend: { labels: { color: '#94a3b8' } }
            }
        }
    });
}

function renderCorrelationMatrix(corrDict) {
    const theadTr = document.getElementById('corr-thead-tr');
    const tbody = document.getElementById('corr-tbody');
    if (!theadTr || !tbody || !corrDict) return;

    const features = Object.keys(corrDict);
    const shortNames = features.map(f => f.replace('_Percentage', '%').replace('_Semester_Marks', ' Marks').replace('_Assessment_Marks', ' Int').replace('_Usage_Hours', ' Hrs'));

    theadTr.innerHTML = '<th class="px-3 py-2 text-left bg-slate-900 text-slate-400">Feature</th>' + shortNames.map(s => `<th class="px-2 py-2 text-center bg-slate-900 text-slate-400">${s}</th>`).join('');

    tbody.innerHTML = features.map((f1, i) => {
        const cells = features.map(f2 => {
            const val = corrDict[f1][f2];
            let bgClass = 'bg-slate-950 text-slate-400';
            if (val === 1.0) bgClass = 'bg-slate-900 text-slate-500 font-bold';
            else if (val > 0.5) bgClass = 'bg-blue-600/40 text-blue-200 font-bold';
            else if (val > 0.2) bgClass = 'bg-blue-500/20 text-blue-300';
            else if (val < -0.2) bgClass = 'bg-red-500/20 text-red-300';
            return `<td class="px-2 py-2 text-center text-xs ${bgClass}">${val}</td>`;
        }).join('');
        return `<tr class="border-b border-slate-800/60"><td class="px-3 py-2 text-left font-semibold text-slate-200">${shortNames[i]}</td>${cells}</tr>`;
    }).join('');
}

window.renderEvaluationCharts = function(metrics, importancesDict) {
    // Model Accuracy Bar Chart
    const accCtx = document.getElementById('chart-accuracy-comp');
    if (accCtx) {
        if (charts.accuracyComp) charts.accuracyComp.destroy();

        const labels = Object.keys(metrics);
        const accs = labels.map(name => metrics[name].accuracy);
        const f1s = labels.map(name => metrics[name].f1_weighted);

        charts.accuracyComp = new Chart(accCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Accuracy (%)', data: accs, backgroundColor: 'rgba(59, 130, 246, 0.85)', borderRadius: 8 },
                    { label: 'F1 Score (%)', data: f1s, backgroundColor: 'rgba(139, 92, 246, 0.85)', borderRadius: 8 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(51, 65, 85, 0.3)' }, ticks: { color: '#94a3b8' }, min: 50, max: 100 }
                },
                plugins: {
                    legend: { labels: { color: '#94a3b8' } }
                }
            }
        });
    }

    // Feature Importances Horizontal Bar Chart
    const impCtx = document.getElementById('chart-feature-importance');
    if (impCtx && importancesDict && importancesDict['Random Forest']) {
        if (charts.featureImportance) charts.featureImportance.destroy();

        const rfImps = importancesDict['Random Forest'];
        const featLabels = rfImps.map(item => item.feature);
        const featValues = rfImps.map(item => (item.importance * 100).toFixed(1));

        charts.featureImportance = new Chart(impCtx, {
            type: 'bar',
            data: {
                labels: featLabels,
                datasets: [{
                    label: 'Random Forest Feature Importance (%)',
                    data: featValues,
                    backgroundColor: 'rgba(16, 185, 129, 0.85)',
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(51, 65, 85, 0.3)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                },
                plugins: {
                    legend: { labels: { color: '#94a3b8' } }
                }
            }
        });
    }
};
