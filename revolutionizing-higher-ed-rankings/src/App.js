import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [data, setData] = useState([]);
  const [categories, setCategories] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedRegion, setSelectedRegion] = useState('all');
  const [checkedCategories, setCheckedCategories] = useState({});
  const [allSelected, setAllSelected] = useState(false);
  const [calculatedScores, setCalculatedScores] = useState([]);
  const rowsPerPage = 25;

  useEffect(() => {
    initialize();
  }, []);

  useEffect(() => {
    if (categories.length > 0) {
      displayRankings();
    }
  }, [currentPage, selectedRegion, checkedCategories, categories, data]);

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
    const loadedData = await loadCSV('university_ranking.csv');
    setData(loadedData);

    // Extract categories dynamically
    if (loadedData.length > 0) {
      const columns = Object.keys(loadedData[0]);
      const extractedCategories = columns.slice(2); // Assume categories start at index 2
      setCategories(extractedCategories);
      
      // Initialize all checkboxes as unchecked
      const initialCheckedState = {};
      extractedCategories.forEach(category => {
        initialCheckedState[category] = false;
      });
      setCheckedCategories(initialCheckedState);
    }
  }

  function displayRankings() {
    const scores = [];
    const seenUniversities = new Set();

    data.forEach(university => {
      const totalScore = categories.reduce((sum, category) => {
        if (checkedCategories[category]) {
          sum += getScore(university.University, category);
        }
        return sum;
      }, 0);

      // Apply region filtering
      if (selectedRegion !== 'all' && university.Continent) {
        const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
        if (!continentMatch) {
          return; // Skip this university if it doesn't match the selected region
        }
      }

      if (!seenUniversities.has(university.University)) {
        scores.push({
          University: university.University,
          Continent: university.Continent || 'Unknown',
          Score: totalScore
        });
        seenUniversities.add(university.University);
      }
    });

    scores.sort((a, b) => b.Score - a.Score);
    setCalculatedScores(scores);
  }

  function getScore(universityName, categoryName) {
    const row = data.find(entry => entry.University === universityName);
    if (!row) return 0;

    const score = parseFloat(row[categoryName]);
    return isNaN(score) ? 0 : score;
  }

  function toggleAllCheckboxes() {
    const newCheckedState = {};
    categories.forEach(category => {
      newCheckedState[category] = !allSelected;
    });
    
    setCheckedCategories(newCheckedState);
    setAllSelected(!allSelected);
    setCurrentPage(1);
  }

  function handleCategoryChange(category) {
    setCheckedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
    setCurrentPage(1);
  }

  function handleRegionChange(e) {
    setSelectedRegion(e.target.value);
    setCurrentPage(1);
  }

  function prevPage() {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  }

  function nextPage() {
    const totalPages = Math.ceil(calculatedScores.length / rowsPerPage);
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  }

  // Get current page data
  const indexOfLastRow = currentPage * rowsPerPage;
  const indexOfFirstRow = indexOfLastRow - rowsPerPage;
  const currentRows = calculatedScores.slice(indexOfFirstRow, indexOfLastRow);
  const totalPages = Math.ceil(calculatedScores.length / rowsPerPage);

  // Filter categories to exclude "Continent"
  const filteredCategories = categories.filter(category => category !== 'Continent');

  return (
    <div className="App">
      <header className="main-header">
        <h1>Revolutionizing Higher Ed Rankings</h1>
        <h2>Improved Scale for Ranking CS Schools</h2>
        <h4>
          <a href="https://github.com/Lianghui818/revolutionizing-higher-ed-rankings">
            GitHub Repository
          </a>
        </h4>
        <h4>
          Rankings are measured through impact that the school's publications have made to the field of computer science. 
          Each impactful publication gets assigned one point. The one point gets divided and assigned to universities 
          according to the authors affiliations.
        </h4>
      </header>
      
      <main>
        <div className="content">
          <div className="filter-container">
            <h3 className="align-left">○ Filter by Area</h3>
            <select 
              id="regionFilter" 
              className="filter-select"
              value={selectedRegion}
              onChange={handleRegionChange}
            >
              <option value="all">All Regions</option>
              <option value="africa">Africa</option>
              <option value="asia">Asia</option>
              <option value="australasia">Australasia</option>
              <option value="europe">Europe</option>
              <option value="north america">North America</option>
              <option value="southamerica">South America</option>
            </select>

            <div className="divider"></div>

            <div className="field-container">
              <h3 className="align-left">○ Filter by Field</h3>
              <div className="button-container">
                <button 
                  id="toggleAll" 
                  className="toggle-button"
                  onClick={toggleAllCheckboxes}
                >
                  {allSelected ? 'Deselect All' : 'Select All'}
                </button>
              </div>
            </div>
            
            <table id="filterTable">
              <tbody>
                {filteredCategories.map((category) => (
                  <tr key={category}>
                    <td>{category}</td>
                    <td>
                      <label className="switch">
                        <input
                          id={category}
                          type="checkbox"
                          checked={checkedCategories[category] || false}
                          onChange={() => handleCategoryChange(category)}
                        />
                        <span className="slider round"></span>
                      </label>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="table-container">
            <table id="rankingTable">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Institution</th>
                  <th>Publication Score</th>
                </tr>
              </thead>
              <tbody>
                {currentRows.map((university, index) => (
                  <tr key={university.University}>
                    <td>{indexOfFirstRow + index + 1}</td>
                    <td>{university.University}</td>
                    <td>{university.Score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            
            <div className="pagination-container">
              <button id="prevPage" onClick={prevPage}>Previous</button>
              <span id="pageIndicator">Page {currentPage} of {totalPages}</span>
              <button id="nextPage" onClick={nextPage}>Next</button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;