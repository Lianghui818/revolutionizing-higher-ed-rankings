// Embed CSV data directly into the JS
const csvData =
`Index,University,Artificial intelligence,Computer systems and networks,Cybersecurity,Databases and data mining,Digital Libraries,Human computer interaction,Machine Learning,Medical Image Computing,Natural Language Processing,Parallel Computing,Program Analysis,Programming Languages,Programming languages and verification,Vision and graphics
1,EPFL,,,,,,,1.0,,,,,,,
2,Google Brain,,,,,,,0.525,,,,,,,
3,University of California,,,,,,,0.4,,,1.1785714285714284,,,,
4,DeepMind,,,,,,,1.1999999999999995,,,,,,,
5,Google Research,,,,,,,0.9999999999999999,,,,,,,
6,Hasso Plattner Institute,,,,,,1.0,,,,,,,,
7,Bell Labs,,,,,,0.6000000000000001,,,,,,,,
8,Telecom ParisTech,,,,,,0.4,,,,,,,,
9,Max Planck Institute for Informatics and Saarland University,,,,,,1.0,,,,,,,,
10,Hasselt University,,,,,,0.5,,,,,,,,
11,Hasselt University-tUL-imec,,,,,,0.25,,,,,,,,
12,Hasselt University - tUL - Flanders Make,,,,,,0.25,,,,,,,,
13,Technische Universität Darmstadt,,,,,,0.8333333333333333,,,,,,,,
14,Max Planck Institute for Informatics,,,,,,0.16666666666666666,,,,,,,,
15,UFF,,,,,,,,,,,,,,0.16666666666666666
16,Institute of Computing,,,,,,,,,,,,,,0.6000000000000001
17,Faculty of Medicine,,,,,,,,,,,,,,0.2
18,Rey Juan Carlos University,,,,,,,,,,,,,,0.2
19,Universidade Federal Fluminense,,,,,,,,,,,,,,0.2857142857142857
20,Universidade do Estado do Rio de Janeiro,,,,,,,,,,,,,,0.5714285714285714
21,Lab. Nacional de Comp. Cientifica,,,,,,,,,,,,,,0.14285714285714285
22,Institute of Computing (IC),,,,,,,,,,,,,,0.5
23,University of B&#x00ED;o-B&#x00ED;o,,,,,,,,,,,,,,0.25
24,University of Lagos,,,,,,,,,,,,,,0.25
25,Université de Montréal,,,,,,,0.875,,,,,,,
26,Let's Encrypt,,,0.08333333333333333,,,,,,,,,,,
27,Cisco,,,0.08333333333333333,,,,,,,,,,,
28,Stanford University,0.25,,0.25,,,,,,,,,,,
29,Electronic Frontier Foundation,,,0.3333333333333333,,,,,,,,,,,
30,University of Michigan,,,0.8666666666666667,,,,,,,,,,,
31,Mozilla,,,0.08333333333333333,,,,,,,,,,,
32,San Jose State University,,,1.0,,,,,,,,,,,
33,CISPA,,,,,,,,,,,1.0,,,
34,National University of Singapore,,,,,,,,,,,1.0,,,
35,KAIST,,,,,,,,,,,0.14285714285714285,,,
36,Boston University,,,,,,,,,,,0.14285714285714285,,,
37,Goldsmiths University of London,,,,,,,,,,,0.14285714285714285,,,
38,Humboldt University of Berlin,,,,,,,,,,,0.2,,,
39,CISPA Helmholtz Center,,,,,,,,,,,0.2,,,
40,Southern University of Science and Technology,,,,,,,,,,,0.5714285714285714,,,
41,Zhejiang University,,,0.2,,,,,,,,0.14285714285714285,,,
42,The University of Hong Kong,,,,,,,,,,,0.14285714285714285,,,
43,University of Illinois,,,,,,,,,,,0.14285714285714285,,,
44,Massachusetts Institute of Technology,,,,,,,,,,0.25,,,,
45,Carnegie Mellon University,,,,,,,,,,1.2023809523809523,,,,
46,MIT CSAIL,,,,0.9999999999999999,,,,,,0.9333333333333333,,,,
47,Northwestern University,0.14285714285714285,1.0,,,,,,,,0.2,,,,
48,University of Maryland,,,,,,,,,,0.5928571428571429,,,,
49,University of California at Riverside,,,,,,,,,,0.5,,,,
50,Microsoft Research Lab India,,,,,,,,,,0.14285714285714285,,,,
51,Safran Electronics &#x0026; Defense,,,1.0,,,,,,,,,,,
52,University of Louisiana at Lafayette,,,1.5000000000000002,,,,,,,,,,,
53,University of Louisiana - Lafayette,,,0.16666666666666666,,,,,,,,,,,
54,MPI-SWS,,,,,,,,,,,,,1.916666666666667,
55,LRI,,,,,,,,,,,,,0.25,
56,Radboud University Nijmegen,,,,,,,,,,,,,0.16666666666666666,
57,University of Cambridge,,,,,,,,,,,,,0.16666666666666666,
58,Université Paris-Saclay - CNRS - ENS Paris-Saclay - Inria,,,,,,,,,,,,,0.16666666666666666,
59,Saarland University,,,,,,,,,,,,,0.16666666666666666,
60,Aarhus University,,,,,,,,,,,,,0.16666666666666666,
61,Microsoft Research,,,,,,,,,,,,,0.2,
62,Seoul National University,,,,,,,,,,,,,0.4,
63,University of Utah,,,,,,,,,,,,,0.4,
64,MBZUAI,0.3333333333333333,,,,,,,,,,,,,
65,Peking University,,3.1071428571428563,,,,,,,,,,,,
66,Stony Brook University,,0.14285714285714285,,,,,,,,,,,,
67,Key Laboratory of High Confidence Software Technologies (Peking University),,0.5,,,,,,,,,,,,
68,School of Computer and Information Technology,,0.25,,,,,,,,,,,,
69,University of Texas at Austin,,0.5,,,,,,,,,,,,
70,Georgia Institute of Technology,,0.3333333333333333,,,,,,,,,,,,
71,Pennsylvania State University,,0.16666666666666666,,,,,,,,,,,,
72,Cornell University,,1.0000000000000002,,,,,,,,,,,,
73,University of Gavle,,1.0,,,,,,,,,,,,
74,Department of Computer Science and Engineering,,,,,,,,1.0,,,,,,
75,University of Salzburg,,,,,,,,0.16666666666666666,,,,,,
76,University of North Carolina,,,,,,,,0.16666666666666666,,,,,,
77,Nanyang Technological University,,,,,,1.0,,,,,,,,
78,Aarhus University in Aarhus,,,,,,1.0,,,,,,,,
79,City University of London,,,,,,1.0,,,,,,,,
80,Northeastern University,,,,,,1.0,,,,,,,,
81,University of Southern California,0.14285714285714285,,,,,,,,,,,,,
82,University of Washington,0.14285714285714285,,,,,,,,,,,,,
83,The Ohio State University,0.25,,,,,,,,,,,,,
84,Uppsala University,0.25,,,,,,,,,,,,,
85,Charles University,0.25,,,,,,,,,,,,,
86,Lucerne University of Applied Sciences,,,,,0.3333333333333333,,,,,,,,,
87,L3S Research Center,,,,,0.3333333333333333,,,,,,,,,
88,TIB - Leibniz Information Centre for Science and Technology,,,,,0.3333333333333333,,,,,,,,,
89,Los Alamos National Lab. (LANL),,,,,0.4,,,,,,,,,
90,National and University Library in Zagreb (Croatia),,,,,0.4,,,,,,,,,
91,University of Zagreb University Computing Centre (Croatia),,,,,0.2,,,,,,,,,
92,Kyoto University,,,,,0.4,,,,,,,,,
93,University of Innsbruck,,,,,0.2,,,,,,,,,
94,AIST,,,,,0.2,,,,,,,,,
95,LIAAD - INESCTEC,,,,,0.2,,,,,,,,,
96,Shizuoka University,,,,,0.5,,,,,,,,,
97,University of Hyogo,,,,,0.5,,,,,,,,,
98,Toshiba Research and Development Center (Japan),,,,0.5,,,,,,,,,,
99,Toshiba Yanazicho Works (Japan),,,,0.5,,,,,,,,,,
100,Stanford,,,,0.375,,,,,,,,,,
101,MIT,,,,0.5,,,,,,,,,,
102,Google,,,,0.0625,,,,,,,,0.3333333333333333,,
103,VMware,,,,0.0625,,,,,,,,,,
104,University of Technology Sydney,,,,,,,,,0.16666666666666666,,,,,
105,Australian National University,,,,,,,,,,,,0.6666666666666666,,
`;


const csvData2 = 
`Index,University,Artificial intelligence,Computer systems and networks,Cybersecurity,Databases and data mining,Digital Libraries,Human computer interaction,Machine Learning,Medical Image Computing,Natural Language Processing,Parallel Computing,Program Analysis,Programming Languages,Programming languages and verification,Vision and graphics
1,EPFL,,,,,,,1.0,,,,,,,
2,Google Brain,,,,,,,0.525,,,,,,,
`;

const csvDataDummy = 
`Index,University,Artificial Intelligence,Machine Learning,Cybersecurity,Bioinformatics,Computer Systems and Networks,Databases and Data Mining,Human Computer Interaction,Vision and Graphics
1,Arizona State University,1000,669,874,800,836,926,977,601
2,Georgia State University,950,990,970,636,828,889,886,984
3,Stanford University,900,618,870,775,635,686,860,885
4,Purdue University,880,735,628,744,600,893,739,712
5,Johns Hopkins University,875,860,650,670,882,986,851,729
6,Duke University,860,691,680,859,879,979,710,790
7,Oregon State University,855,996,823,603,603,834,660,797
8,California Institute of Technology,830,886,619,911,788,936,884,714
9,Elon University,820,701,753,792,954,648,942,887
10,New York University,805,926,644,954,961,806,960,705
11,Drexel University,795,784,646,702,970,680,954,751
12,Boston University,785,744,870,693,776,736,928,656
13,University of Florida,770,727,896,765,760,729,956,621
14,Virginia Tech,755,822,880,813,734,919,989,770
15,Rice University,740,963,709,851,608,970,1000,679
16,Baylor University,725,688,827,975,722,745,740,782
17,University of Denver,710,785,877,890,940,963,773,760
18,Vanderbilt University,695,869,736,645,791,908,895,822
19,University of Georgia,680,918,899,743,889,900,900,683
20,Princeton University,665,918,782,980,896,992,894,838
`;

// Process CSV data
const rows = csvData.trim().split('\n');
const headers = rows.shift().split(',');
const data = rows.map(row => {
	const rowData = row.split(',');
	return headers.reduce((obj, header, index) => {
		obj[header.trim()] = rowData[index].trim();
		return obj;
	}, {});
});

var categories = [];
console.log(data);

function displayFilters() { //dynamically creating filter table 
    var table = document.getElementById('filterTable');
    var tableBody = table.getElementsByTagName('tbody')[0];
    tableBody.innerHTML = ''; // Clear existing rows
    const columns = Object.keys(data[0]);

    for (let index=2; index<columns.length; index++) {
        categories.push(columns[index]);
        const row = tableBody.insertRow();
        row.innerHTML = `
                    <td>${columns[index]}</td>
                    <td> 
                        <label class="switch">
                            <input id="${columns[index]}" type="checkbox" onclick="displayRankings()">
                            <span class="slider round"></span>
                        </label>
                    </td>`;
        
    };
}


function toggleAllCheckboxes() {
    var allChecked = true;
    var checkboxes = document.querySelectorAll('#filterTable input[type="checkbox"]');

    checkboxes.forEach(function (checkbox) {    // Not all boxes checked
        if (!checkbox.checked) {
            allChecked = false;
        }
    });

    checkboxes.forEach(function (checkbox) {    
        checkbox.checked = !allChecked;
    });

    var toggleButton = document.getElementById('toggleAll');
    toggleButton.textContent = allChecked ? 'Select All' : 'Deselect All';

    displayRankings();
}


function displayRankings() {
    let calculatedScores = [];
    data.forEach(university => {
        var totalScore = 0; 
        categories.forEach(categoryName => {
            var selected = document.getElementById(categoryName).checked;
            if (selected === true) {
                //totalScore = totalScore + parseInt(university[categoryName]);
                totalScore += getScore(university.University, categoryName);
            }
        });
        const universityData = {University:university.University, score:totalScore};
       calculatedScores.push(universityData);

    });
    calculatedScores = calculatedScores.sort((a, b) => {
          if (a.score > b.score) {
            return -1;
          }
    });
    var table = document.getElementById('rankingTable');
    var tableBody = table.getElementsByTagName('tbody')[0];
    tableBody.innerHTML = ''; // Clear existing rows
            var rank = 1;
            calculatedScores.forEach(university => {
                const row = tableBody.insertRow();
                row.innerHTML = `
                    <td>${rank}</td>
                    <td>${university.University}</td>
                    <td>${university.score}</td>`;
                rank++;
            });
}

function getScore(universityName, categoryName) {
    // Find the row corresponding to the university
    const row = data.find(entry => entry.University === universityName);

    if (!row) {
        console.log(`University '${universityName}' not found.`);
        return null;
    }

    // Return the score for the given category
    let score = row[categoryName];

    if (score === undefined) {
        console.log(`Category '${categoryName}' not found.`);
        return null;
    }
    
    console.log(`University: '{$universityName}', Category '${categoryName}', Score '${score}'`)

    // If the score is blank or undefined, return 0
    if (score === "") {
        return 0;
    }

    // Parse score as an integer
    const parsedScore = parseFloat(score);
    return isNaN(parsedScore) ? 0 : parsedScore;
}


function getScoreOld(universityName, categoryName) {
    // Find the row corresponding to the university
    const row = data.find(entry => entry.University === universityName);
    
    if (!row) {
        console.log(`University '${universityName}' not found.`);
        return null;
    }

    // Return the score for the given category
    const score = row[categoryName];
    
    if (score === undefined) {
        console.log(`Category '${categoryName}' not found.`);
        return null;
    }
    return parseInt(score); // Parse score as integer
}

// Example usage
const universityName = "Stanford University";
const categoryName = "Artificial Intelligence";
const score = getScore(universityName, categoryName);
displayFilters();

if (score !== null) {
    console.log(`Score for ${categoryName} at ${universityName}: ${score}`);
}
