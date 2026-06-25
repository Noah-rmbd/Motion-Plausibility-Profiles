let plotlyShapesCache = {};
// Maps transition_id -> shape index, and observation_id -> shape index in the Plotly shapes array
let shapeIndexMap = {};

async function renderPlotlyProfile(user_id) {
    try {
        shapeIndexMap = {}; // Reset shape-to-index mapping for new user

        const userTransitions = appData.transitions_list.filter(t => String(t.user_id) === String(user_id));
        const userObscured = appData.obscured_observations.filter(o => String(o.user_id) === String(user_id));
        const userObservations = appData.observations.filter(o => String(o.user_id) === String(user_id));

        let transitionObsIds = new Set();
        for (let t of userTransitions) {
            transitionObsIds.add(String(t.observation_id1));
            transitionObsIds.add(String(t.observation_id2));
        }
        let isolatedObs = userObservations.filter(o => !transitionObsIds.has(String(o.observation_id)));

        let shapeList = [];
        let markerSize = 8.0;
        let dataEntryCounter = 0;
        let validDatesPlotly = new Set();

        // Group transitions by date
        let transitionsByDate = {};
        for (let t of userTransitions) {
            let d = t.date.trim();
            if (!transitionsByDate[d]) transitionsByDate[d] = [];
            transitionsByDate[d].push(t);
            validDatesPlotly.add(d.replace(/\//g, '-'));
        }

        // Group obscured by date
        let obscuredByDate = {};
        for (let o of userObscured) {
            let d = o.date.trim();
            if (!obscuredByDate[d]) obscuredByDate[d] = [];
            obscuredByDate[d].push(o);
            validDatesPlotly.add(d.replace(/\//g, '-'));
        }

        // Group isolated by date
        let isolatedByDate = {};
        for (let o of isolatedObs) {
            let d = o.date.trim().substring(0, 10).replace(/-/g, '/');
            if (!isolatedByDate[d]) isolatedByDate[d] = [];
            isolatedByDate[d].push(o);
            validDatesPlotly.add(d.replace(/\//g, '-'));
        }

        // Get all unique dates sorted
        let allDates = Array.from(new Set([...Object.keys(transitionsByDate), ...Object.keys(obscuredByDate), ...Object.keys(isolatedByDate)])).sort();

        for (let date of allDates) {
            dataEntryCounter += 1;
            let horizontalOffset = 0.0;

            let dateAnchor = date.replace(/\//g, '-');
            let trans = transitionsByDate[date] || [];
            let obs = obscuredByDate[date] || [];
            let isolated = isolatedByDate[date] || [];

            // Draw obscured shapes first (as squares)
            for (let o of obs) {
                let shapeIdx = shapeList.length;
                shapeIndexMap['obs_' + String(o.observation_id)] = shapeIdx;
                shapeList.push({
                    type: "rect",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: { color: 'black', width: 1 },
                    fillcolor: 'white'
                });
                horizontalOffset += markerSize;
            }

            // Draw isolated shapes as green circles
            for (let o of isolated) {
                let shapeIdx = shapeList.length;
                shapeIndexMap['obs_' + String(o.observation_id)] = shapeIdx;
                shapeList.push({
                    type: "circle",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: { width: 0 },
                    fillcolor: 'rgb(26, 150, 65)' // Green for isolated unobscured observations
                });
                horizontalOffset += markerSize;
            }

            // Draw first unobscured point if trans exists
            if (trans.length > 0) {
                let firstShapeIdx = shapeList.length;
                // Tag with the first observation_id of the first transition
                shapeIndexMap['obs_' + String(trans[0].observation_id1)] = firstShapeIdx;
                shapeList.push({
                    type: "circle",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: { width: 0 },
                    fillcolor: 'rgb(128,128,128)' // First point is grey
                });
                horizontalOffset += markerSize;
            }

            // Draw remaining transitions
            for (let t of trans) {
                let additionalOffset = t.distance >= 0 ? Math.log2(t.distance + 1.0) : 0;
                horizontalOffset += additionalOffset;

                let fillcolorBySpeed = getSpeedColor(t.speed + " km/h");
                if (t.elapsed_time === 0) fillcolorBySpeed = 'rgb(0,0,0)'; // instantaneous

                let tShapeIdx = shapeList.length;
                // Tag with transition_id and also second observation_id
                shapeIndexMap['trans_' + String(t.transition_id)] = tShapeIdx;
                shapeIndexMap['obs_' + String(t.observation_id2)] = tShapeIdx;
                const isUnplausible = t.is_unplausible === true;
                const isReviewedPlausible = t.reviewed_plausible === true;
                shapeList.push({
                    type: "circle",
                    xsizemode: 'pixel', ysizemode: 'pixel',
                    xanchor: dateAnchor,
                    yanchor: dataEntryCounter,
                    x0: markerSize * 0.5 + horizontalOffset, y0: -markerSize * 0.5,
                    x1: markerSize * 1.5 + horizontalOffset, y1: markerSize * 0.5,
                    line: isUnplausible
                        ? { color: '#ff4d4f', width: 2.5 }
                        : (isReviewedPlausible ? { color: '#00a676', width: 2.5 } : { width: 0 }),
                    fillcolor: fillcolorBySpeed,
                    isUnplausible: isUnplausible,
                    isReviewedPlausible: isReviewedPlausible
                });
                horizontalOffset += markerSize;
            }
        }

        plotlyShapesCache[user_id] = shapeList;

        const dummyX = shapeList.map(s => s.xanchor);
        const dummyY = shapeList.map(s => s.yanchor);
        const hoverInfo = dummyX.map(x => validDatesPlotly.has(x) ? 'all' : 'skip');

        const dummyData = [{
            x: dummyX,
            y: dummyY,
            mode: 'markers',
            marker: { opacity: 0 },
            hoverinfo: hoverInfo
        }];

        let layout = {
            margin: { l: 80, r: 10, b: 40, t: 10, pad: 4 },
            font: { size: 10 },
            showlegend: false,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {
                tickangle: 0,
                tickformat: '%Y',
                gridwidth: 0.5,
                gridcolor: 'rgb(230,230,230)',
            },
            yaxis: {
                range: [0, dataEntryCounter + 1.5],
                zeroline: true,
                zerolinewidth: 1,
                zerolinecolor: '#2a3f5f',
                gridwidth: 0.5,
                gridcolor: 'rgb(230,230,230)',
            },
            shapes: shapeList
        };

        Plotly.newPlot('mppPlotly', dummyData, layout, { responsive: true, scrollZoom: true });

        document.getElementById('mppPlotly').on('plotly_click', function (data) {
            if (data.points && data.points.length > 0) {
                const clickedDateString = data.points[0].x;
                if (!validDatesPlotly.has(clickedDateString)) return;

                const clickedDate = clickedDateString.replace(/-/g, '/');

                if (selectedDate.includes(clickedDate)) {
                    selectedDate = selectedDate.filter(d => d !== clickedDate);
                } else {
                    selectedDate.push(clickedDate);
                }
                updatePlausibleLabelAction();
                console.log(`Plotly point clicked! Date: ${clickedDate}`);

                const allDays = document.querySelectorAll('.timeline-day');
                allDays.forEach(el => {
                    if (selectedDate.includes(el.textContent)) {
                        el.classList.add('selected');
                    } else {
                        el.classList.remove('selected');
                    }
                });

                updateMapForDate(user_id, selectedDate);
            }
        });

    } catch (error) {
        console.error('Error rendering Plotly Profile:', error);
    }
}

function highlightPlotlyDay(user_id, dates) {
    if (!plotlyShapesCache[user_id]) return;
    const shapes = plotlyShapesCache[user_id];
    const formattedDates = dates.map(d => d.replace(/\//g, '-'));

    // Create a new array of shapes to trigger relayout
    const updatedShapes = shapes.map(shape => {
        if (formattedDates.length === 0) {
            return shape; // Return to initial state
        } else if (formattedDates.includes(shape.xanchor)) {
            let border = { width: 0 };
            if (shape.fillcolor === 'white' || shape.fillcolor === 'rgb(255,255,255)') {
                border = { ...shape.line, color: 'black', width: 2 };
            } else if (shape.isUnplausible) {
                border = { color: '#ff4d4f', width: 2.5 };
            } else if (shape.isReviewedPlausible) {
                border = { color: '#00a676', width: 2.5 };
            } else {
                border = { color: 'rgb(50,50,50)', width: 2 };
            }
            return {
                ...shape,
                opacity: 1,
                line: border
            };
        } else {
            return { ...shape, opacity: 0.15 };
        }
    });

    try {
        Plotly.relayout('mppPlotly', { shapes: updatedShapes });
    } catch (e) {
        console.error("Error highlighting Plotly day:", e);
    }
}

// Highlight a single shape on the Plotly profile when hovering a map element
let _lastHighlightedShapeIdx = null;
function highlightProfileShape(shapeIdx) {
    if (!currentUser || !plotlyShapesCache[currentUser]) return;
    if (_lastHighlightedShapeIdx === shapeIdx) return; // avoid redundant relayouts
    _lastHighlightedShapeIdx = shapeIdx;

    const shapes = plotlyShapesCache[currentUser];
    const formattedDates = selectedDate.map(d => d.replace(/\//g, '-'));

    const updatedShapes = shapes.map((shape, idx) => {
        let s = { ...shape };
        // Keep day-based dimming
        if (formattedDates.length > 0) {
            s.opacity = formattedDates.includes(shape.xanchor) ? 1 : 0.15;
        }
        // Add highlight ring on the hovered shape
        if (idx === shapeIdx) {
            s.line = { color: 'rgba(255, 0, 0, 1)', width: 3 };
        } else if (s.isUnplausible) {
            s.line = { color: '#ff4d4f', width: 2.5 };
        } else if (s.isReviewedPlausible) {
            s.line = { color: '#00a676', width: 2.5 };
        }
        return s;
    });

    try {
        Plotly.relayout('mppPlotly', { shapes: updatedShapes });
    } catch (e) { /* ignore */ }
}

function clearProfileHighlight() {
    if (_lastHighlightedShapeIdx === null) return;
    _lastHighlightedShapeIdx = null;
    // Re-apply the normal day highlighting
    if (currentUser) {
        highlightPlotlyDay(currentUser, selectedDate);
    }
}
