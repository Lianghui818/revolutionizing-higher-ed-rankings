let data = [];
let facultyData = [];
let categories = [];
let selectedRegion = 'all';
let lastFiltered = [];
let selectedCountry = 'all';
let expandedRows = new Set();

const ACTIVE_FIELDS = [
    "Machine Learning",
    "Computer Vision & Image Processing",
    "Natural Language Processing",
];

const DISPLAY_LABELS = {
    'Machine Learning': 'Machine Learning',
    'Computer Vision & Image Processing': 'Computer Vision & Image Processing',
    'Natural Language Processing': 'Natural Language Processing',
};

const EPS = 1e-9;
const NORMALIZE = 'unit-variance';
let fieldStats = {};

async function loadCSV(filePath) {
    const response = await fetch(filePath);
    const csvText = await response.text();
    
    const rows = csvText.trim().split('\n');
    const headers = rows.shift().split(',');

    return rows.map(row => {
        const rowData = row.split(',');
        return headers.reduce((obj, header, index) => {
            obj[header.trim()] = rowData[index]?.trim() || '';
            return obj;
        }, {});
    });
}

async function initialize() {
    data = await loadCSV('3_f_1.csv');
    facultyData = await loadCSV('3_faculty_score.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    generateFieldCheckboxes();

    setupFieldFilter();

    computeFieldStats();
    setupRegionFilter();
    setupCountryFilter();
    displayRankings();
}

function setupCountryFilter() {
    const sel = document.getElementById('countryFilter');
    const countries = Array.from(new Set(
        data.map(r => (r.Country || '').trim()).filter(s => s && s.toLowerCase() !== 'unknown')
    )).sort((a, b) => a.localeCompare(b));

    sel.innerHTML = '<option value="all">All Countries</option>' +
        countries.map(c => `<option value="${c}">${c}</option>`).join('');

    sel.addEventListener('change', () => {
        selectedCountry = sel.value;
        resetPageAndDisplayRankings();
    });
}

function computeFieldStats() {
    fieldStats = {};
    const cols = ACTIVE_FIELDS;
    cols.forEach(cat => {
        const vals = data.map(r => parseFloat(r[cat])).filter(v => !isNaN(v) && v > 0);
        const n = vals.length;
        const mean = n ? vals.reduce((a,b)=>a+b,0)/n : 0;
        const varSample = n>1 ? vals.reduce((a,b)=>a+(b-mean)*(b-mean),0)/(n-1) : 0;
        const std = Math.sqrt(varSample);
        // fieldStats[cat] = { mean, std };
        // only use mean here:
        fieldStats[cat] = { mean };
    });
}

function setupFieldFilter() {
    // Add event listeners to all field checkboxes
    container.addEventListener('change', (e) => {
        if (e.target && e.target.classList.contains('field-checkbox')) {
          updateToggleAllFieldsButton();
          resetPageAndDisplayRankings(); 
        }
      });

      updateToggleAllFieldsButton();
}

function toggleAllFields() {
    const checkboxes = document.querySelectorAll('.field-checkbox');
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    
    checkboxes.forEach(checkbox => {
        checkbox.checked = !allChecked;
    });
    
    updateToggleAllFieldsButton();
    resetPageAndDisplayRankings();
}

function updateToggleAllFieldsButton() {
    const checkboxes = document.querySelectorAll('.field-checkbox');
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    const button = document.getElementById('toggleAllFields');
    
    if (button) {
        button.textContent = allChecked ? 'None' : 'All';
    }
}

function toId(name) {
    return 'field-' + name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
}

function updateToggleAllButtonLabel() {
    const boxes = document.querySelectorAll('.field-checkbox');
    const allChecked = Array.from(boxes).every(b => b.checked);
    const btn = document.getElementById('toggleAll');
    if (btn) btn.textContent = allChecked ? 'Deselect All' : 'Select All';
}

function setupRegionFilter() {
    const regionFilter = document.getElementById('regionFilter');
    regionFilter.addEventListener('change', resetPageAndDisplayRankings);
}

function resetPageAndDisplayRankings() {
    selectedRegion = document.getElementById('regionFilter').value;
    expandedRows.clear(); // Clear expanded rows when filters change
    displayRankings();
}

function generateFieldCheckboxes() {
    const container = document.getElementById('fieldCheckboxContainer');
    container.innerHTML = '';
    
    ACTIVE_FIELDS.forEach(field => {
        const displayLabel = DISPLAY_LABELS[field] || field;
        const checkboxItem = document.createElement('label');
        checkboxItem.className = 'field-checkbox-item';
        checkboxItem.innerHTML = `
        <input type="checkbox" class="field-checkbox" data-field="${field}" checked>
        <span class="checkbox-label">${displayLabel}</span>
        `;

        container.appendChild(checkboxItem);
    });
}

function getSelectedCategories() {
    return [...document.querySelectorAll('.field-checkbox:checked')]
           .map(el => el.dataset.field);
}

function getRawScore(univ, field) {
    const row = data.find(e => e.University === univ);
    if (!row) return 0;
    const s = parseFloat(row[field]);
    return isNaN(s) ? 0 : s;
}

function getNormalizedScore(univ, field) {
    const s = getRawScore(univ, field);
    if (!(s > 0)) return 0;
    const stats = fieldStats[field] || { std: 1 };
    // const std = stats.std > EPS ? stats.std : 1;
    // return s / std;
    // use mean
    const mean = stats.mean > EPS ? stats.mean : 1;
    return s / mean;
}

function getFacultyForUniversity(universityName, selectedCategories) {
    // Filter faculty data for this university and selected categories
    const filteredFaculty = facultyData.filter(faculty => {
        const matchesUniversity = faculty.University === universityName;
        const matchesCategory = selectedCategories.length === 0 || 
                                selectedCategories.includes(faculty.Category);
        return matchesUniversity && matchesCategory;
    });

    // Group by faculty name and sum scores
    // change to: Group by faculty name and calculate normalized scores
    const facultyMap = new Map();
    
    filteredFaculty.forEach(faculty => {
        const name = faculty['Faculty Name'] || 'Unknown';
        const score = parseFloat(faculty.Score) || 0;
        const category = faculty.Category || 'Unknown';
        
        if (!facultyMap.has(name)) {
            facultyMap.set(name, {
                name: name,
                categories: [],
                totalScore: 0
            });
        }
        
        const facultyInfo = facultyMap.get(name);
        facultyInfo.totalScore += score;
        if (!facultyInfo.categories.includes(category)) {
            facultyInfo.categories.push(category);
        }
    });
    
    // Convert to array and sort by total score
    const mergedFaculty = Array.from(facultyMap.values())
        .sort((a, b) => b.totalScore - a.totalScore);
    
    return mergedFaculty;
}

function toggleUniversityDropdown(universityName, rowElement) {
    const isExpanded = expandedRows.has(universityName);
    
    if (isExpanded) {
        // Collapse: Remove the details row and any chart stats
        const detailsRow = rowElement.nextElementSibling;
        if (detailsRow && detailsRow.classList.contains('faculty-details-row')) {
            detailsRow.remove();
        }
        // Also remove chart stats if present
        const chartRow = rowElement.nextElementSibling;
        if (chartRow && chartRow.classList.contains('chart-stats-row')) {
            chartRow.remove();
        }
        expandedRows.delete(universityName);
        rowElement.querySelector('.expand-icon').innerHTML = '▶';
    } else {
        // Expand: Add the details row
        const selectedCategories = getSelectedCategories();
        const facultyList = getFacultyForUniversity(universityName, selectedCategories);
        
        const detailsRow = document.createElement('tr');
        detailsRow.classList.add('faculty-details-row');
        
        let facultyHTML = '<td colspan="3"><div class="faculty-details">';
        
        if (facultyList.length > 0) {
            facultyHTML += '<table class="faculty-table">';
            
            // Adjust header based on number of selected categories
            const fieldHeader = selectedCategories.length === 1 ? 'Field' : 'Fields';
            facultyHTML += `<thead><tr><th>Faculty Name</th><th>${fieldHeader}</th><th>Total Contribution</th></tr></thead>`;
            facultyHTML += '<tbody>';
            
            facultyList.forEach(faculty => {
                const fieldsDisplay = faculty.categories.join(', ');
                facultyHTML += `
                    <tr>
                        <td>${faculty.name}</td>
                        <td>${fieldsDisplay}</td>
                        <td>${faculty.totalScore.toFixed(4)}</td>
                    </tr>
                `;
            });
            
            facultyHTML += '</tbody></table>';
        } else {
            facultyHTML += '<p>No faculty data available for the selected fields.</p>';
        }
        
        facultyHTML += '</div></td>';
        detailsRow.innerHTML = facultyHTML;
        
        rowElement.insertAdjacentElement('afterend', detailsRow);
        expandedRows.add(universityName);
        rowElement.querySelector('.expand-icon').innerHTML = '▼';
    }
}

function displayRankings() {
    showLoadingSpinner();
    
    // Simulate loading delay for better UX
    setTimeout(() => {
        const calculatedScores = [];
        const seenUniversities = new Set();

        data.forEach(university => {
            const selectedCats = getSelectedCategories();

            let totalScore = 0;
            if (selectedCats.length >= 2) {
                // Normalized harmonic mean for multiple fields
                let denom = 0, count = 0;
                selectedCats.forEach(cat => {
                    const x = getNormalizedScore(university.University, cat);
                    if (x > 0) { 
                        denom += 1 / (x + EPS); 
                        count++;
                    }
                });
                totalScore = count > 0 ? (count / denom) : 0;
            } else if (selectedCats.length === 1) {
                // Raw score for single field
                totalScore = getRawScore(university.University, selectedCats[0]);
            } else {
                totalScore = 0;
            }

            // Apply region filtering
            if (selectedRegion !== 'all' && university.Continent) {
                const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
                if (!continentMatch) return;
            }
            
            // Apply countries filtering
            if (selectedCountry !== 'all') {
                const country = (university.Country || '').trim();
                if (!country || country !== selectedCountry) return;
            }

            if (!seenUniversities.has(university.University) && totalScore > 0) {
                calculatedScores.push({
                    University: university.University,
                    Continent: university.Continent || 'Unknown',
                    Score: totalScore
                });
                seenUniversities.add(university.University);
            }
        });

        calculatedScores.sort((a, b) => b.Score - a.Score);

        lastFiltered = calculatedScores;
        updateStats(calculatedScores);
        displayAllRankings(calculatedScores);
        hideLoadingSpinner();
    }, 300);
}

function displayAllRankings(data) {
    const table = document.getElementById('rankingTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = '';

    data.forEach((university, index) => {
        const row = tableBody.insertRow();
        row.classList.add('university-row', 'fade-in');
        
        const rank = index + 1;
        const flagClass = getFlagClass(university.Continent);
        const chartIcon = generateChartIcon(university.University);
        
        const universityCell = `
            <td class="rank-col">${rank}</td>
            <td class="institution-col university-name-cell">
                <span class="expand-icon">▶</span>
                <span class="university-name" title="View details" onclick="toggleUniversityDropdown('${university.University.replace(/'/g, "\\'")}', this.closest('tr'))">${university.University}</span>
                <span class="flag-icon ${flagClass}"></span>
                ${chartIcon}
            </td>
            <td class="score-col">
                <span class="score-value">${university.Score.toFixed(2)}</span>
            </td>
        `;
        
        row.innerHTML = universityCell;
        
        // Re-expand if this university was previously expanded
        if (expandedRows.has(university.University)) {
            setTimeout(() => {
                toggleUniversityDropdown(university.University, row);
            }, 0);
        }
    });
}

// Removed pagination functions - now using scroll mode

// New utility functions
function showLoadingSpinner() {
    document.getElementById('loadingSpinner').style.display = 'flex';
}

function hideLoadingSpinner() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function updateStats(data) {
    const totalUniversities = data.length;
    const totalScore = data.reduce((sum, uni) => sum + uni.Score, 0);
    const activeFilters = getActiveFilterCount();
    
    document.getElementById('totalUniversities').textContent = totalUniversities.toLocaleString();
    document.getElementById('totalScore').textContent = totalScore.toFixed(1);
    document.getElementById('activeFilters').textContent = activeFilters;
    
    // Update scroll info
    document.getElementById('totalCount').textContent = totalUniversities.toLocaleString();
}

function getActiveFilterCount() {
    let count = 0;
    if (selectedRegion !== 'all') count++;
    if (selectedCountry !== 'all') count++;
    count += getSelectedCategories().length;
    return count;
}

function getFlagClass(continent) {
    const flagMap = {
        'North America': 'flag-us',
        'Europe': 'flag-gb',
        'Asia': 'flag-cn',
        'Africa': 'flag-default',
        'Australasia': 'flag-au',
        'South America': 'flag-default'
    };
    return flagMap[continent] || 'flag-default';
}

function getGlobalMaxScore(selectedCats) {
    let globalMax = 0;
    
    // Find the maximum score across all universities for the selected categories
    data.forEach(university => {
        selectedCats.forEach(field => {
            const score = getRawScore(university.University, field);
            if (score > globalMax) {
                globalMax = score;
            }
        });
    });
    
    return globalMax;
}

function getTopFields(universityName, chartData, globalMaxScore) {
    const topFields = [];
    
    chartData.forEach(item => {
        // Check if this university has the highest score in this field
        const isTop = item.score === globalMaxScore && item.score > 0;
        if (isTop) {
            // Get the display name for the field
            const displayName = getFieldDisplayName(item.field);
            topFields.push(displayName);
        }
    });
    
    return topFields;
}

function getFieldDisplayName(field) {
    // const displayNames = {
    //     'Machine Learning': 'Machine Learning',
    //     'Computer Vision & Image Processing': 'Computer Vision',
    //     'Natural Language Processing': 'Natural Language Processing'
    // };
    // return displayNames[field] || field;
    return DISPLAY_LABELS[field] || field;
}

function generateChartIcon(universityName) {
    return `<i class="fas fa-chart-bar chart-icon" onclick="toggleChartStats('${universityName.replace(/'/g, "\\'")}', this.closest('tr'))" title="View field statistics"></i>`;
}

function toggleChartStats(universityName, row) {
    // Check if chart is already expanded (look in next sibling row)
    const nextRow = row.nextElementSibling;
    if (nextRow && nextRow.classList.contains('chart-stats-row')) {
        nextRow.remove();
        return;
    }
    
    const university = data.find(u => u.University === universityName);
    if (!university) return;
    
    const selectedCats = getSelectedCategories();
    const chartData = selectedCats.map(field => ({
        field: field,
        score: getRawScore(universityName, field)
    })).filter(item => item.score > 0);
    
    if (chartData.length === 0) return;
    
    // Create chart row
    const chartRow = document.createElement('tr');
    chartRow.classList.add('chart-stats-row');
    
    // Use global maximum score for consistent scaling
    const globalMaxScore = getGlobalMaxScore(selectedCats);
    
    // Check for top performers
    const topFields = getTopFields(universityName, chartData, globalMaxScore);
    
    let chartHTML = '<td colspan="3"><div class="chart-stats-container">';
    chartHTML += '<h4>Field Statistics</h4>';
    
    // Add top performer notice if any
    if (topFields.length > 0) {
        chartHTML += '<div class="top-performer-notice">';
        chartHTML += '<i class="fas fa-trophy"></i>';
        chartHTML += '<span>Top of ' + topFields.join(', ') + '</span>';
        chartHTML += '</div>';
    }
    
    chartHTML += '<div class="chart-scale">';
    chartHTML += '<span class="scale-label">Scale: 0 - ' + globalMaxScore.toFixed(2) + '</span>';
    chartHTML += '</div>';
    chartHTML += '<div class="chart-bars">';
    
    chartData.forEach(item => {
        const percentage = (item.score / globalMaxScore) * 100;
        chartHTML += `
            <div class="chart-bar-item">
                <div class="chart-bar-label">${item.field}</div>
                <div class="chart-bar-container">
                    <div class="chart-bar" style="width: ${percentage}%"></div>
                    <div class="chart-bar-value">${item.score.toFixed(2)}</div>
                </div>
            </div>
        `;
    });
    
    chartHTML += '</div></div></td>';
    chartRow.innerHTML = chartHTML;
    
    // Insert after current row
    row.parentNode.insertBefore(chartRow, row.nextSibling);
}

function exportData() {
    if (lastFiltered.length === 0) {
        alert('No data to export');
        return;
    }
    
    const csvContent = [
        ['Rank', 'University', 'Continent', 'Impact Score'],
        ...lastFiltered.map((uni, index) => [
            index + 1,
            uni.University,
            uni.Continent,
            uni.Score.toFixed(2)
        ])
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-rankings-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function showAbout() {
    alert('AI Research Impact Rankings\n\nThis ranking system measures academic excellence through research influence and impact, using LLM analysis to identify the most important references in academic papers.\n\nBuilt with ❤️ for academic transparency.');
}

function toggleDemoNotice() {
    const notice = document.querySelector('.demo-notice');
    notice.style.display = 'none';
    document.querySelector('.main-header').style.marginTop = '0';
}

// Initialize the application
initialize();