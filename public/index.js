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

// only use mean to do normalization，no variance/std
function computeFieldStats() {
    fieldStats = {};
    const cols = ACTIVE_FIELDS;
    cols.forEach(cat => {
        const vals = data.map(r => parseFloat(r[cat])).filter(v => !isNaN(v) && v > 0);
        const n = vals.length;
        const mean = n ? vals.reduce((a,b)=>a+b,0)/n : 0;
        fieldStats[cat] = { mean };
    });
}

function setupFieldFilter() {
    const container = document.getElementById('fieldCheckboxContainer');
    
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

// use mean
function getNormalizedScore(univ, field) {
    const s = getRawScore(univ, field);
    if (!(s > 0)) return 0;
    const stats = fieldStats[field] || { mean: 1 };
    const mean = stats.mean > EPS ? stats.mean : 1;
    return s / mean;
}

// normalization faculy score
function getFacultyForUniversity(universityName, selectedCategories) {
    // Filter faculty data for this university and selected categories
    const filteredFaculty = facultyData.filter(faculty => {
        const matchesUniversity = faculty.University === universityName;
        const matchesCategory = selectedCategories.length === 0 || 
                                selectedCategories.includes(faculty.Category);
        return matchesUniversity && matchesCategory;
    });

    // Group by faculty name and calculate normalized scores
    const facultyMap = new Map();
    
    filteredFaculty.forEach(faculty => {
        const name = faculty['Faculty Name'] || 'Unknown';
        const rawScore = parseFloat(faculty.Score) || 0;        //raw score here
        const category = faculty.Category || 'Unknown';
        
        // normalization
        const stats = fieldStats[category] || { mean: 1 };
        const mean = stats.mean > EPS ? stats.mean : 1;
        const normalizedScore = rawScore / mean;
        
        if (!facultyMap.has(name)) {
            facultyMap.set(name, {
                name: name,
                categories: [],
                totalScore: 0,
                categoryScores: {} // every category score after normalized
            });
        }
        
        const facultyInfo = facultyMap.get(name);
        facultyInfo.totalScore += normalizedScore; // add normalized score
        if (!facultyInfo.categories.includes(category)) {
            facultyInfo.categories.push(category);
        }
        
        if (!facultyInfo.categoryScores[category]) {
            facultyInfo.categoryScores[category] = 0;
        }
        facultyInfo.categoryScores[category] += normalizedScore;
    });
    
    // Convert to array and sort by total normalized score
    const mergedFaculty = Array.from(facultyMap.values())
        .sort((a, b) => b.totalScore - a.totalScore);
    
    return mergedFaculty;
}

// expand-icon clickable
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
        const expandIcon = rowElement.querySelector('.expand-icon');
        if (expandIcon) {
            expandIcon.innerHTML = '▶';
            expandIcon.classList.remove('expanded');
        }
    } else {
        // Expand: Add the details row
        const selectedCategories = getSelectedCategories();
        const faculty = getFacultyForUniversity(universityName, selectedCategories);
        
        if (faculty.length === 0) {
            return;
        }
        
        const detailsRow = document.createElement('tr');
        detailsRow.classList.add('faculty-details-row');
        
        // show normaliz_score
        let facultyHTML = '<td colspan="3"><div class="faculty-details"><table class="faculty-table">';
        facultyHTML += '<thead><tr><th>Faculty Name</th><th>Fields</th><th>Normalized Score</th></tr></thead>';
        facultyHTML += '<tbody>';
        
        faculty.forEach(f => {
            const categories = f.categories.join(', ');
            facultyHTML += `
                <tr>
                    <td>${f.name}</td>
                    <td>${categories}</td>
                    <td>${f.totalScore.toFixed(2)}</td>
                </tr>
            `;
        });
        
        facultyHTML += '</tbody></table></div></td>';
        detailsRow.innerHTML = facultyHTML;
        
        // Insert after current row
        rowElement.parentNode.insertBefore(detailsRow, rowElement.nextSibling);
        expandedRows.add(universityName);
        const expandIcon = rowElement.querySelector('.expand-icon');
        if (expandIcon) {
            expandIcon.innerHTML = '▼';
            expandIcon.classList.add('expanded');
        }
    }
}

function displayRankings() {
    showLoadingSpinner();
    
    setTimeout(() => {
        const selectedCats = getSelectedCategories();
        
        if (selectedCats.length === 0) {
            const tableBody = document.getElementById('rankingTable').querySelector('tbody');
            tableBody.innerHTML = '<tr><td colspan="3" style="text-align: center; padding: 2rem; color: var(--gray-500);">Please select at least one research field</td></tr>';
            updateStats([]);
            hideLoadingSpinner();
            return;
        }
        
        let filtered = data.filter(d => {
            const continent = (d.Continent || '').trim();
            const country = (d.Country || '').trim();
            
            const regionMatch = selectedRegion === 'all' || continent === selectedRegion;
            const countryMatch = selectedCountry === 'all' || country === selectedCountry;
            
            return regionMatch && countryMatch;
        });
        
        // use normaliz_score to compute
        const calculatedScores = filtered.map(university => {
            const score = selectedCats.reduce((sum, field) => {
                return sum + getNormalizedScore(university.University, field);
            }, 0);
            
            return {
                University: university.University,
                Continent: university.Continent,
                Country: university.Country,
                Score: score
            };
        }).filter(u => u.Score > 0)
          .sort((a, b) => b.Score - a.Score);

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
        const flagClass = getFlagClass(university.Country);
        const chartIcon = generateChartIcon(university.University);
        
        const universityCell = `
            <td class="rank-col">${rank}</td>
            <td class="institution-col university-name-cell">
                <span class="expand-icon" onclick="toggleUniversityDropdown('${university.University.replace(/'/g, "\\'")}', this.closest('tr'))" title="Expand/Collapse">▶</span>
                <span class="university-name" onclick="toggleUniversityDropdown('${university.University.replace(/'/g, "\\'")}', this.closest('tr'))" title="View details">${university.University}</span>
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

function getFlagClass(country) {
    const flagMap = {
        'United States': 'flag-us',
        'USA': 'flag-us',
        'Canada': 'flag-ca',
        
        'United Kingdom': 'flag-gb',
        'UK': 'flag-gb',
        'Germany': 'flag-de',
        'France': 'flag-fr',
        
        'China': 'flag-cn',
        'Japan': 'flag-jp',
        
        'Australia': 'flag-au',
    };
    return flagMap[country] || 'flag-default';
}

function getGlobalMaxScore(selectedCats) {
    let globalMax = 0;
    
    // Find the maximum normalized score across all universities for the selected categories
    data.forEach(university => {
        selectedCats.forEach(field => {
            const score = getNormalizedScore(university.University, field);
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
    return DISPLAY_LABELS[field] || field;
}

function generateChartIcon(universityName) {
    return `<i class="fas fa-chart-bar chart-icon" onclick="toggleChartStats('${universityName.replace(/'/g, "\\'")}', this.closest('tr'))" title="View field statistics"></i>`;
}

// show normalized score in chart
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
        score: getNormalizedScore(universityName, field)
    })).filter(item => item.score > 0);
    
    if (chartData.length === 0) return;
    
    // Create chart row
    const chartRow = document.createElement('tr');
    chartRow.classList.add('chart-stats-row');
    
    // Use global maximum normalized score for consistent scaling
    const globalMaxScore = getGlobalMaxScore(selectedCats);
    
    // Check for top performers
    const topFields = getTopFields(universityName, chartData, globalMaxScore);
    
    let chartHTML = '<td colspan="3"><div class="chart-stats-container">';
    chartHTML += '<h4>Field Statistics (Normalized)</h4>';
    
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
        ['Rank', 'University', 'Continent', 'Normalized Impact Score'],
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