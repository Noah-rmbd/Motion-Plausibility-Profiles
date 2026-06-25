async function setTimeline(user_id) {
    try {
        const activeDays = new Set();

        // Find all days with unobscured transitions for this user
        appData.transitions_list.forEach(t => {
            if (String(t.user_id) === String(user_id) && t.date.includes('/')) {
                activeDays.add(t.date.trim());
            }
        });

        // Add days with obscured observations
        appData.obscured_observations.forEach(o => {
            if (String(o.user_id) === String(user_id)) {
                let d = o.date.trim().substring(0, 10).replace(/-/g, '/');
                activeDays.add(d);
            }
        });

        globalDaysArray = Array.from(activeDays).sort();
        console.log(`User ${user_id} has observations on ${globalDaysArray.length} unique days.`);

        renderTimeline();

        return globalDaysArray;

    } catch (error) {
        console.error('Error generating timeline:', error);
        return [];
    }
}

// --- Timeline Framework ---

function buildTimelineTokens(mode, daysArray) {
    if (!daysArray || daysArray.length === 0) return [];

    if (mode === 'day') {
        return daysArray.map(day => ({ label: day, days: [day] }));
    }
    else if (mode === 'month') {
        const monthsMap = {};
        daysArray.forEach(day => {
            const parts = day.split('/');
            if (parts.length >= 2) {
                const monthLabel = parts[0] + '/' + parts[1];
                if (!monthsMap[monthLabel]) monthsMap[monthLabel] = [];
                monthsMap[monthLabel].push(day);
            }
        });
        return Object.keys(monthsMap).sort().map(month => ({ label: month, days: monthsMap[month] }));
    }
    else if (mode === 'year') {
        const yearsMap = {};
        daysArray.forEach(day => {
            const parts = day.split('/');
            if (parts.length >= 1) {
                const yearLabel = parts[0];
                if (!yearsMap[yearLabel]) yearsMap[yearLabel] = [];
                yearsMap[yearLabel].push(day);
            }
        });
        return Object.keys(yearsMap).sort().map(year => ({ label: year, days: yearsMap[year] }));
    }
    else if (mode === 'smart') {
        // This mode clusters observations that were done in maximum 5 days in a row, and returns the 5 most important clusters
        let clusters = [];
        let currentCluster = { label: '', days: [] };
        let prevDate = null;

        daysArray.forEach(day => {
            const currDate = new Date(day.replace(/\//g, '-'));
            if (!prevDate) {
                currentCluster.days.push(day);
            } else {
                const diffTime = Math.abs(currDate - prevDate);
                const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

                if (diffDays > 5) {
                    clusters.push(currentCluster);
                    currentCluster = { label: '', days: [day] };
                } else {
                    currentCluster.days.push(day);
                }
            }
            prevDate = currDate;
        });
        if (currentCluster.days.length > 0) {
            clusters.push(currentCluster);
        }

        clusters.sort((a, b) => b.days.length - a.days.length);
        const topClusters = clusters.slice(0, 5);
        topClusters.sort((a, b) => a.days[0].localeCompare(b.days[0]));

        return topClusters.map(cluster => {
            const startNode = cluster.days[0];
            const endNode = cluster.days[cluster.days.length - 1];
            return {
                label: startNode === endNode ? startNode : `${startNode} to ${endNode}`,
                days: cluster.days
            };
        });
    }
    return [];
}

function renderTimeline() {
    const timelineList = document.getElementById('timeline');
    const modeSelect = document.getElementById('timeline-mode');
    if (!timelineList || !modeSelect) return;

    timelineList.innerHTML = '';
    const mode = modeSelect.value;
    const tokens = buildTimelineTokens(mode, globalDaysArray);

    tokens.forEach(token => {
        const li = document.createElement('li');
        li.className = 'timeline-day';
        li.textContent = token.label;

        const tokenTrajectories = appData.trajectories_list.filter(traj =>
            String(traj.user_id) === String(currentUser)
            && token.days.includes(traj.date.trim())
        );
        const hasBaselineAnomaly = tokenTrajectories.some(traj => traj.baseline_unplausible);
        const hasModelAnomaly = tokenTrajectories.some(traj => traj.model_unplausible);
        const hasReviewedPlausible = tokenTrajectories.some(traj => traj.reviewed_plausible);
        const baselineCount = tokenTrajectories.filter(
            traj => traj.baseline_unplausible
        ).length;
        const modelCount = tokenTrajectories.filter(
            traj => traj.model_unplausible
        ).length;
        const plausibleCount = tokenTrajectories.filter(
            traj => traj.reviewed_plausible
        ).length;
        li.title = `${baselineCount} physical-rule violation(s), ${modelCount} model detection(s), ${plausibleCount} reviewed plausible`;
        if (hasBaselineAnomaly && hasModelAnomaly) {
            li.classList.add('combined-anomaly');
        } else if (hasBaselineAnomaly) {
            li.classList.add('baseline-anomaly');
        } else if (hasModelAnomaly) {
            li.classList.add('model-anomaly');
        } else if (hasReviewedPlausible) {
            li.classList.add('plausible-label');
        }

        const isFullySelected = token.days.length > 0 && token.days.every(d => selectedDate.includes(d));
        if (isFullySelected) {
            li.classList.add('selected');
        }

        li.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDraggingTimeline = true;

            const currentlySelected = token.days.every(d => selectedDate.includes(d));
            if (currentlySelected) {
                selectedDate = selectedDate.filter(d => !token.days.includes(d));
                li.classList.remove('selected');
            } else {
                token.days.forEach(d => {
                    if (!selectedDate.includes(d)) selectedDate.push(d);
                });
                li.classList.add('selected');
            }
            updatePlausibleLabelAction();
        });

        li.addEventListener('mouseenter', () => {
            if (isDraggingTimeline) {
                const currentlySelected = token.days.every(d => selectedDate.includes(d));
                if (currentlySelected) {
                    selectedDate = selectedDate.filter(d => !token.days.includes(d));
                    li.classList.remove('selected');
                } else {
                    token.days.forEach(d => {
                        if (!selectedDate.includes(d)) selectedDate.push(d);
                    });
                    li.classList.add('selected');
                }
                updatePlausibleLabelAction();
            }
        });

        timelineList.appendChild(li);
    });
}
