document.addEventListener('mouseup', () => {
    if (isDraggingTimeline) {
        isDraggingTimeline = false;
        if (currentUser) {
            updateMapForDate(currentUser, selectedDate);
        }
    }
});

async function setRightPanelView(view) {
    const panel = document.getElementById('chart-container');
    const views = {
        mpp: document.getElementById('mpp-panel-view'),
        stats: document.getElementById('stats-panel')
    };
    const tabs = {
        mpp: document.getElementById('right-panel-mpp-tab'),
        stats: document.getElementById('right-panel-stats-tab')
    };
    if (!panel || !views[view] || !tabs[view]) return;
    if (!panel.classList.contains('hidden') && tabs[view].classList.contains('active')) {
        closeRightPanel();
        return;
    }
    panel.classList.remove('hidden');
    Object.entries(views).forEach(([name, element]) => {
        const active = name === view;
        element.classList.toggle('active', active);
        tabs[name].classList.toggle('active', active);
        tabs[name].setAttribute('aria-selected', String(active));
    });
    if (view === 'mpp') {
        syncMppEmptyState();
        setTimeout(() => {
            if (window.Plotly) Plotly.Plots.resize('mppPlotly');
        }, 0);
    } else if (view === 'stats') {
        try {
            await loadDatasetStats();
            setTimeout(() => {
                if (window.Plotly) Plotly.Plots.resize('stats-plot');
            }, 0);
        } catch (error) {
            document.getElementById('stats-summary').textContent = error.message;
        }
    }
    setTimeout(() => {
        if (leafletMap) leafletMap.invalidateSize();
    }, 0);
}

function closeRightPanel() {
    const panel = document.getElementById('chart-container');
    const btn = document.getElementById('toggle-profile');
    if (!panel) return;
    panel.classList.add('hidden');
    document.querySelectorAll('[data-panel-view]').forEach(tab => {
        tab.classList.remove('active');
        tab.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.right-panel-view').forEach(view => {
        view.classList.remove('active');
    });
    if (btn) btn.textContent = '◀';
    setTimeout(() => {
        if (leafletMap) leafletMap.invalidateSize();
    }, 0);
}

function setLeftPanelView(view) {
    const panel = document.getElementById('left-panel');
    const views = {
        model: document.getElementById('left-panel-model-view'),
        map: document.getElementById('left-panel-map-view')
    };
    const tabs = {
        model: document.getElementById('left-panel-model-tab'),
        map: document.getElementById('left-panel-map-tab')
    };
    if (!panel || !views[view] || !tabs[view]) return;
    if (!panel.hidden && tabs[view].classList.contains('active')) {
        closeLeftPanel();
        return;
    }
    panel.hidden = false;
    Object.entries(views).forEach(([name, element]) => {
        const active = name === view;
        element.classList.toggle('active', active);
        tabs[name].classList.toggle('active', active);
        tabs[name].setAttribute('aria-selected', String(active));
    });
    setTimeout(() => {
        if (leafletMap) leafletMap.invalidateSize();
    }, 0);
}

function closeLeftPanel() {
    const panel = document.getElementById('left-panel');
    if (!panel) return;
    panel.hidden = true;
    document.querySelectorAll('[data-left-panel-view]').forEach(tab => {
        tab.classList.remove('active');
        tab.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.left-panel-view').forEach(view => {
        view.classList.remove('active');
    });
    setTimeout(() => {
        if (leafletMap) leafletMap.invalidateSize();
    }, 0);
}

function syncMppEmptyState() {
    const empty = document.getElementById('mpp-empty-state');
    const plot = document.getElementById('mppPlotly');
    if (!empty || !plot) return;
    const needsUser = !currentUser;
    empty.hidden = !needsUser;
    plot.hidden = needsUser;
}

// Automatically load and populate the user-select dropdown when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    // 0. Initialize the Map
    leafletMap = L.map('map', { zoomControl: false }).setView([0, 0], 2); // Default to a global view
    L.control.zoom({ position: 'topright' }).addTo(leafletMap);

    // Log zoom level and hide arrows when zoomed out to improve performance
    function checkZoomLevel() {
        const currentZoom = leafletMap.getZoom();
        console.log(`Current map zoom level: ${currentZoom}`);
        if (currentZoom < 6) {
            leafletMap.getContainer().classList.add('hide-arrows');
        } else {
            leafletMap.getContainer().classList.remove('hide-arrows');
        }
    }
    leafletMap.on('zoomend', checkZoomLevel);
    leafletMap.on('zoomend moveend', () => {
        renderTrajectoryBubbles();
    });
    checkZoomLevel(); // Initial check

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(leafletMap);

    // 0.5 Build Legend dynamically
    let legendControl = null;
    function updateLegend() {
        if (legendControl) leafletMap.removeControl(legendControl);

        legendControl = L.control({ position: 'bottomright' });
        legendControl.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'info legend');
            const p = isColorblindMode ? colorPaletteColorblind : colorPaletteStandard;
            const categories = [
                { threshold: '0 km/h', color: p.c0 },
                { threshold: '< 5 km/h', color: p.c5 },
                { threshold: '< 10 km/h', color: p.c10 },
                { threshold: '< 25 km/h', color: p.c25 },
                { threshold: '< 80 km/h', color: p.c80 },
                { threshold: '< 200 km/h', color: p.c200 },
                { threshold: '>= 200 km/h', color: p.cMax }
            ];
            div.innerHTML = '<strong>Speed Key</strong><br>';
            for (let i = 0; i < categories.length; i++) {
                div.innerHTML +=
                    '<i style="background-color:' + categories[i].color + '; display: inline-block; width: 50px; height: 16px; float: left; margin-right: 8px; margin-top: 3px; border-radius: 3px; border: 1px solid rgba(0, 0, 0, 0.2);"></i> ' +
                    categories[i].threshold + '<br>';
            }
            return div;
        };
        legendControl.addTo(leafletMap);
    }

    updateLegend();
    initializeTrajectoryBubbleControls();

    // Colorblind mode toggle listener
    const colorblindToggle = document.getElementById('colorblind-toggle');
    if (colorblindToggle) {
        colorblindToggle.addEventListener('change', (e) => {
            isColorblindMode = e.target.checked;
            updateLegend();

            // Clear Plotly cache to force recalculation of shape colors
            plotlyShapesCache = {};

            // Re-render visuals if a user is already active
            if (currentUser) {
                renderPlotlyProfile(currentUser);
                if (selectedDate) {
                    updateMapForDate(currentUser, selectedDate);
                }
            }
        });
    }

    // Timeline resolution toggle listener
    const timelineModeSelect = document.getElementById('timeline-mode');
    if (timelineModeSelect) {
        timelineModeSelect.addEventListener('change', () => {
            renderTimeline();
            updatePlausibleLabelAction();
        });
    }

    /*const plausibleButton = document.getElementById('mark-plausible-button');
    if (plausibleButton) {
        plausibleButton.addEventListener('click', markSelectedTrajectoriesPlausible);
        updatePlausibleLabelAction();
    }*/

    let usernames = {};
    try {
        const statsResponse = await fetch('ressources/usernames.json');
        if (statsResponse.ok) {
            usernames = await statsResponse.json();
        }
    } catch (e) { console.warn("Could not load usernames"); }

    const selectElement = document.getElementById('user-select');
    if (!selectElement) {
        console.warn('Could not find the element with ID "user-select"');
        return;
    }

    // Add a 'change' event listener to the <select> element itself
    selectElement.addEventListener('change', (event) => {
        const selectedUserId = event.target.value;
        if (selectedUserId) {
            setUser(selectedUserId);
        }
        refreshTrajectoryBubblesForFilter();
        syncMppEmptyState();
    });

    async function initializeDataset() {
        // Clear UI state
        currentUser = null;
        selectedDate = [];
        syncMppEmptyState();
        expandedBubbleLayers.forEach(layer => layer.remove());
        expandedBubbleLayers = [];
        if (leafletMap) {
            currentMarkers.forEach(m => leafletMap.removeLayer(m));
            currentMarkers = [];
            if (window.routePolyline) leafletMap.removeLayer(window.routePolyline);
            clearTrajectoryBubbleLayer();
            clearSelectedTrajectoryLayers();
        }
        document.getElementById('timeline').innerHTML = '';
        Plotly.purge('mppPlotly');

        selectElement.innerHTML = '<option value="">--Please choose a user--</option>';

        const users = await fetchAndParseUsers();

        let userCounts = {};
        if (appData && appData.users_list) {
            for (const u of appData.users_list) {
                userCounts[String(u.username)] = u.nb_trajectories || u.trajectory_count || 0;
            }
        }

        users.sort((a, b) => {
            const countA = userCounts[a] || 0;
            const countB = userCounts[b] || 0;
            return countB - countA;
        });

        const topUsers = users.slice(0, 20000);
        const fragment = document.createDocumentFragment();

        for (const user of topUsers) {
            const option = document.createElement('option');
            option.value = user;
            // For Gowalla, there are no usernames, just user IDs. We check currentDatabase.
            const uname = (currentDatabase === 'original' && usernames[user]) ? usernames[user] :
                (currentDatabase === 'synthetic' ? user.replace(/_/g, ' ') : `User ${user}`);
            const count = userCounts[user] || 0;
            option.textContent = `${uname} (${count} trajectories)`;
            fragment.appendChild(option);
        }
        selectElement.appendChild(fragment);
        refreshTrajectoryBubblesForFilter();
    }

    const databaseSelect = document.getElementById('database-select');
    if (databaseSelect) {
        databaseSelect.addEventListener('change', (e) => {
            currentDatabase = e.target.value;
            scoreDistributionPayload = null;
            appliedTrajectoryScoreFilter = null;
            updateModelDropdown();
            initializeDataset();
        });
    }

    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', async (e) => {
            currentModelSelection = e.target.value;
            scoreDistributionPayload = null;
            appliedTrajectoryScoreFilter = null;
            refreshModelPanels();
            syncTrajectoryFilterControls();
            scheduleScoreDistributionRefresh();
            if (currentUser) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
            }
        });
    }

    const thresholdSlider = document.getElementById('threshold-slider');
    if (thresholdSlider) {
        let thresholdTimer = null;
        thresholdSlider.addEventListener('input', async (e) => {
            document.getElementById('threshold-percentage').textContent = `${parseFloat(e.target.value).toFixed(1)}%`;
            updatePerformanceMetrics();
            scoreDistributionPayload = null;
            appliedTrajectoryScoreFilter = null;
            scheduleScoreDistributionRefresh();
            clearTimeout(thresholdTimer);
            thresholdTimer = setTimeout(async () => {
                if (currentUser) {
                    await fetchModelScoresForUser(currentUser);
                    setTimeline(currentUser);
                    renderPlotlyProfile(currentUser);
                }
            }, 100);
        });
    }

    [
        'aggregation-select',
        'transition-score-select',
        'first-transition-select',
    ].forEach(id => {
        document.getElementById(id).addEventListener('change', async () => {
            refreshModelPanels();
            scoreDistributionPayload = null;
            appliedTrajectoryScoreFilter = null;
            scheduleScoreDistributionRefresh();
            if (currentUser) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
            }
        });
    });
    const transitionFeatureOptions = document.getElementById('transition-feature-options');
    if (transitionFeatureOptions) {
        transitionFeatureOptions.addEventListener('change', async (event) => {
            const options = [...transitionFeatureOptions.querySelectorAll('input[type="checkbox"]')];
            if (!options.some(option => option.checked) && event.target) {
                event.target.checked = true;
            }
            refreshModelPanels();
            scoreDistributionPayload = null;
            appliedTrajectoryScoreFilter = null;
            scheduleScoreDistributionRefresh();
            if (currentUser) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
            }
        });
    }

    const leftPanel = document.getElementById('left-panel');
    const closeLeftPanelButton = document.getElementById('close-left-panel');
    document.querySelectorAll('[data-left-panel-view]').forEach(button => {
        button.addEventListener('click', () => setLeftPanelView(button.dataset.leftPanelView));
    });
    document.querySelectorAll('[data-panel-view]').forEach(button => {
        button.addEventListener('click', () => setRightPanelView(button.dataset.panelView));
    });
    closeLeftPanelButton.addEventListener('click', () => {
        closeLeftPanel();
    });
    ['stats-dataset', 'stats-subset', 'stats-metric'].forEach(id => {
        document.getElementById(id).addEventListener('change', async () => {
            try {
                await loadDatasetStats();
            } catch (error) {
                document.getElementById('stats-summary').textContent = error.message;
            }
        });
    });
    document.getElementById('export-stats-json').addEventListener('click', () => {
        if (!currentStatsPayload) return;
        const name = `${currentStatsPayload.dataset}_${currentStatsPayload.subset}_${currentStatsPayload.metric}`;
        downloadStatsFile(`${name}.json`, JSON.stringify(currentStatsPayload, null, 2), 'application/json');
    });
    document.getElementById('export-stats-csv').addEventListener('click', () => {
        if (!currentStatsPayload) return;
        const rows = ['bin_left,bin_right,count,comparison_count'].concat(
            currentStatsPayload.histogram.map((row, index) =>
                `${row.bin_left},${row.bin_right},${row.count},${currentStatsPayload.comparison_histogram?.[index]?.count ?? ''}`
            )
        );
        const name = `${currentStatsPayload.dataset}_${currentStatsPayload.subset}_${currentStatsPayload.metric}`;
        downloadStatsFile(`${name}.csv`, rows.join('\n'), 'text/csv');
    });
    document.getElementById('export-stats-png').addEventListener('click', () => {
        if (!currentStatsPayload) return;
        Plotly.downloadImage('stats-plot', {
            format: 'png',
            filename: `${currentStatsPayload.dataset}_${currentStatsPayload.subset}_${currentStatsPayload.metric}`,
            width: 1200,
            height: 700
        });
    });
    [
        'filter-model-type',
        'filter-training-mode',
        'filter-feature-representation'
    ].forEach(id => {
        document.getElementById(id).addEventListener('change', async () => {
            updateModelDropdown();
            syncTrajectoryFilterControls();
            scoreDistributionPayload = null;
            appliedTrajectoryScoreFilter = null;
            scheduleScoreDistributionRefresh();
            if (currentUser && currentModelSelection) {
                await fetchModelScoresForUser(currentUser);
                setTimeline(currentUser);
                renderPlotlyProfile(currentUser);
            }
        });
    });

    // Initial load. Query parameters are also useful for reproducible visual checks.
    loadAvailableModels().then(async () => {
        populateModelFilters();
        updateModelDropdown();
        const params = new URLSearchParams(window.location.search);
        const requestedDataset = params.get('dataset');
        const requestedUser = params.get('user');
        if (requestedDataset) {
            currentDatabase = requestedDataset === 'inat' ? 'original' : requestedDataset;
            databaseSelect.value = currentDatabase;
            updateModelDropdown();
            if (currentDatabase === 'synthetic') {
                setLeftPanelView('model');
            }
        }
        await initializeDataset();
        if (requestedUser) {
            selectElement.value = requestedUser;
            if (selectElement.value) await setUser(requestedUser);
        }
        syncMppEmptyState();
    });
});

function toggleProfile() {
    const chart = document.getElementById('chart-container');
    const btn = document.getElementById('toggle-profile');
    if (chart.classList.contains('hidden')) {
        setRightPanelView('mpp');
        chart.style.width = '';
        if (btn) btn.textContent = '▶';
        setTimeout(() => {
            if (window.Plotly) Plotly.Plots.resize('mppPlotly');
            if (window.Plotly) Plotly.Plots.resize('stats-plot');
            if (leafletMap) leafletMap.invalidateSize();
        }, 300);
    } else {
        closeRightPanel();
    }
}

// Resizer logic
document.addEventListener('DOMContentLoaded', () => {
    const resizer = document.getElementById('resizer');
    const chart = document.getElementById('chart-container');
    const container = document.getElementById('content-split');

    if (!resizer || !chart || !container) return;

    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        document.body.style.webkitUserSelect = 'none';
        chart.style.transition = 'none'; // Prevent CSS transition lagging during manual resize
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const containerRect = container.getBoundingClientRect();
        let newWidth = containerRect.right - e.clientX;
        const minWidth = 250;
        const isHidden = chart.classList.contains('hidden');

        if (isHidden) {
            // Panel is currently closed. Only open it once the drag width crosses minWidth.
            if (newWidth >= minWidth) {
                setRightPanelView('mpp');
                if (newWidth > containerRect.width - 200) newWidth = containerRect.width - 200;
                chart.style.width = newWidth + 'px';
                chart.style.flex = 'none';
                document.getElementById('toggle-profile').textContent = '▶';
                if (window.Plotly) Plotly.Plots.resize('mppPlotly');
                if (window.Plotly) Plotly.Plots.resize('stats-plot');
                if (leafletMap) leafletMap.invalidateSize();
            }
        } else {
            // Panel is open. If dragging below minWidth, collapse it.
            if (newWidth < minWidth) {
                closeRightPanel();
            } else {
                if (newWidth > containerRect.width - 200) newWidth = containerRect.width - 200;
                chart.style.width = newWidth + 'px';
                chart.style.flex = 'none';

                if (window.Plotly) Plotly.Plots.resize('mppPlotly');
                if (window.Plotly) Plotly.Plots.resize('stats-plot');
                if (leafletMap) leafletMap.invalidateSize();
            }
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.body.style.webkitUserSelect = '';
            chart.style.transition = ''; // Restore transition
            if (leafletMap) leafletMap.invalidateSize();
        }
    });
});
