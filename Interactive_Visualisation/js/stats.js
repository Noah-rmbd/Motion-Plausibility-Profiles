const statsMetricMetadata = {
    speed: { label: 'Speed', unit: 'km/h' },
    acceleration: { label: 'Acceleration', unit: 'm/s²' },
    distance: { label: 'Distance', unit: 'm' },
    elapsed_time: { label: 'Elapsed time', unit: 's' },
    bearing_change: { label: 'Bearing change', unit: 'rad' },
    trajectory_n_transitions: { label: 'Transitions per trajectory', unit: 'transitions' },
    max_speed: { label: 'Maximum trajectory speed', unit: 'km/h' },
    total_distance: { label: 'Total trajectory distance', unit: 'm' },
    total_elapsed_time: { label: 'Total trajectory elapsed time', unit: 's' },
    avg_elapsed_time: { label: 'Average transition elapsed time', unit: 's' },
    model_score: { label: 'Model anomaly score', unit: 'score' }
};

function formatStatNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '--';
    if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function downloadStatsFile(filename, content, type) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([content], { type }));
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
}

function updateStatsControlStates() {
    const subset = document.getElementById('stats-subset').value;
    const metricSelect = document.getElementById('stats-metric');
    if (!metricSelect) return;

    const isTransitionPopulation = subset === 'transition_flagged' || subset === 'transition_accepted';
    const isModelPopulation = [
        'model_flagged',
        'model_accepted',
        'transition_flagged',
        'transition_accepted'
    ].includes(subset);

    const trajectoryMetrics = ['trajectory_n_transitions', 'max_speed', 'total_distance', 'total_elapsed_time', 'avg_elapsed_time', 'model_score'];
    const modelOnlyMetrics = ['max_speed', 'total_distance', 'total_elapsed_time', 'model_score'];

    Array.from(metricSelect.options).forEach(option => {
        const val = option.value;
        if (trajectoryMetrics.includes(val)) {
            if (isTransitionPopulation) {
                option.disabled = true;
            } else if (!isModelPopulation && modelOnlyMetrics.includes(val)) {
                option.disabled = true;
            } else {
                option.disabled = false;
            }
        } else {
            option.disabled = false;
        }
    });

    const selectedOption = metricSelect.options[metricSelect.selectedIndex];
    if (selectedOption && selectedOption.disabled) {
        metricSelect.value = 'speed';
    }
}

async function loadDatasetStats() {
    updateStatsControlStates();
    const dataset = document.getElementById('stats-dataset').value;
    const subset = document.getElementById('stats-subset').value;
    const metric = document.getElementById('stats-metric').value;
    const isModelPopulation = [
        'model_flagged',
        'model_accepted',
        'transition_flagged',
        'transition_accepted'
    ].includes(subset);
    let response;
    if (isModelPopulation) {
        if (!currentModelSelection) {
            throw new Error('Select a model before loading model population statistics.');
        }
        response = await fetch('/api/model_stats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: currentModelSelection,
                dataset,
                population: subset,
                metric,
                anomaly_percentage: parseFloat(
                    document.getElementById('threshold-slider').value
                ),
                aggregation: document.getElementById('aggregation-select').value,
                transition_score_mode:
                    document.getElementById('transition-score-select').value,
                transition_score_features: selectedTransitionScoreFeatures(),
                include_first_transition:
                    document.getElementById('first-transition-select').value === 'true'
            })
        });
    } else {
        response = await fetch(`/api/stats?dataset=${dataset}&subset=${subset}&metric=${metric}`);
    }
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Statistics request failed');
    }
    currentStatsPayload = await response.json();
    renderDatasetStats(currentStatsPayload);
}

function renderDatasetStats(payload) {
    const metadata = statsMetricMetadata[payload.metric];
    const summary = payload.summary;
    const quantiles = summary.quantiles || {};
    const isTransitionScope = payload.subset === 'transition_flagged' || payload.subset === 'transition_accepted';
    const scopeLabel = isTransitionScope ? 'Scope: Step Level (Transition)' : 'Scope: Sequence Level (Trajectory)';
    const scopeClass = isTransitionScope ? 'badge-transition' : 'badge-trajectory';

    let modelContext = '';
    if (payload.model_id) {
        if (Number.isFinite(Number(payload.flagged_transitions))) {
            modelContext = `
                <dt>Flagged transitions</dt><dd>${payload.flagged_transitions.toLocaleString()} (${formatMetric(payload.flagged_transition_rate)})</dd>
                <dt>Threshold</dt><dd>${formatModelScore(payload.threshold)} transition score</dd>`;
        } else {
            modelContext = `
                <dt>Flagged trajectories</dt><dd>${payload.flagged_trajectories.toLocaleString()} (${formatMetric(payload.flagged_rate)})</dd>
                <dt>Threshold</dt><dd>${formatModelScore(payload.threshold)} trajectory score</dd>`;
        }
    }
    document.getElementById('stats-summary').innerHTML = `
        <div class="stats-scope-container">
            <span class="stats-scope-badge ${scopeClass}">${scopeLabel}</span>
        </div>
        <dl>
            <dt>Trajectories</dt><dd>${payload.n_trajectories.toLocaleString()}</dd>
            <dt>Transitions</dt><dd>${payload.n_transitions.toLocaleString()}</dd>
            <dt>Mean</dt><dd>${formatStatNumber(summary.mean)} ${metadata.unit}</dd>
            <dt>Median</dt><dd>${formatStatNumber(quantiles.p50)} ${metadata.unit}</dd>
            <dt>P95</dt><dd>${formatStatNumber(quantiles.p95)} ${metadata.unit}</dd>
            <dt>P99</dt><dd>${formatStatNumber(quantiles.p99)} ${metadata.unit}</dd>
            ${modelContext}
        </dl>`;

    const scaleType = document.getElementById('stats-scale')?.value || 'linear';
    const isLogScale = scaleType === 'log';

    const traces = [{
        type: 'bar',
        name: payload.subset === 'model_flagged' ? 'Model flagged' : (
            payload.subset === 'model_accepted' ? 'Model accepted' : (
                payload.subset === 'transition_flagged' ? 'Flagged transitions' : (
                    payload.subset === 'transition_accepted' ? 'Accepted transitions' : 'Selected population'
                )
            )
        ),
        x: payload.histogram.map(bin => (bin.bin_left + bin.bin_right) / 2),
        y: payload.histogram.map(bin => (isLogScale ? (bin.count > 0 ? bin.count : null) : bin.count)),
        width: payload.histogram.map(bin => bin.bin_right - bin.bin_left),
        marker: { color: '#287d5b' },
        hovertemplate: `%{x:.4g} ${metadata.unit}<br>%{y:,} rows<extra></extra>`
    }];
    if (payload.comparison_histogram && payload.comparison_histogram.length) {
        traces.unshift({
            type: 'bar',
            name: payload.comparison_label,
            x: payload.comparison_histogram.map(bin => (bin.bin_left + bin.bin_right) / 2),
            y: payload.comparison_histogram.map(bin => (isLogScale ? (bin.count > 0 ? bin.count : null) : bin.count)),
            width: payload.comparison_histogram.map(bin => bin.bin_right - bin.bin_left),
            marker: { color: '#9aa2a0', opacity: 0.48 },
            hovertemplate: `%{x:.4g} ${metadata.unit}<br>%{y:,} rows<extra></extra>`
        });
    }
    Plotly.react('stats-plot', traces, {
        margin: { l: 58, r: 18, t: 34, b: 52 },
        title: { text: `${metadata.label} distribution`, font: { size: 14 } },
        xaxis: { title: metadata.unit, fixedrange: false },
        yaxis: {
            title: isLogScale ? 'Count (log scale)' : 'Count',
            type: isLogScale ? 'log' : 'linear',
            autorange: true,
            fixedrange: false
        },
        bargap: 0.02,
        barmode: 'overlay',
        paper_bgcolor: 'white',
        plot_bgcolor: '#f7f8f7'
    }, {
        responsive: true,
        displaylogo: false
    });

    const featureErrors = payload.feature_errors || [];
    document.getElementById('stats-feature-errors').innerHTML = featureErrors.length
        ? `
            <h3>Feature reconstruction errors</h3>
            <table class="metric-table">
                <thead><tr><th>Feature</th><th>Mean</th><th>vs all</th></tr></thead>
                <tbody>
                    ${featureErrors.slice(0, 8).map(row => `
                        <tr>
                            <td>${profileDisplayName(row.feature)}</td>
                            <td>${formatModelScore(row.mean_error)}</td>
                            <td>${Number.isFinite(row.lift) ? `${row.lift.toFixed(2)}×` : '--'}</td>
                        </tr>`).join('')}
                </tbody>
            </table>`
        : '';

    const quality = payload.data_quality || {};
    document.getElementById('stats-quality').innerHTML = Object.keys(quality).length
        ? `
            <h3>Data quality</h3>
            <dl>
                <dt>Undefined acceleration</dt><dd>${Number(quality.undefined_acceleration || 0).toLocaleString()}</dd>
                <dt>Undefined bearing</dt><dd>${Number(quality.undefined_bearing_change || 0).toLocaleString()}</dd>
                <dt>Stationary with turn</dt><dd>${Number(quality.stationary_nonzero_bearing_change || 0).toLocaleString()}</dd>
                <dt>Zero elapsed time</dt><dd>${Number(quality.zero_elapsed_time || 0).toLocaleString()}</dd>
            </dl>`
        : '';
}
