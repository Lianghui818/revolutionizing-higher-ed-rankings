# AI Rankings

**By:**  
- Lianghui Wang: LLM / Frontend / Backend
- Karl Mellinger: LLM
- Baorong Luo: Frontend 
- Jack Herbst: Frontend  
- Andrew Ketola: Backend
- Mainoah Muna: Backend  
- Diji Yang: PhD student from UCSC
- Dr. Yi Zhang: Professor from UCSC
- Dr. Fuxin Li: Associate Professor from Oregon State University

---

## Motivation

CSRankings have significantly improved the university ranking system for computer science domains over earlier ones such as USNews by focusing on objective measures such as paper publication. However, one peril is that when CSRankings are used, rankings have become a numbers game where more publications are simply rewarded with higher rankings. This is dangerous for the field as it encourages researchers to chase **quantity vs. quality** in their publications.  

---

## Our Approach

**How can we measure the quality of publications?**
We believe:
1. The quality of research is best measured by peers in the same research area.
2. Careful use of LLMs can reveal the inherent quality judgments peers have already made through their reading and citation practices.

Hence, we developed a new ranking system that analyzes research papers from major AI conferences.

For each paper, we ask a large language model (DeepSeek-R1-Distill-Llama-8B):
**What are the five most important papers to this paper?**
In other words, the five works that most strongly influence the study. By doing this, we trace which papers and authors are consistently regarded as foundational to new discoveries.

Next, we map these influential authors to their affiliated universities using the CSRankings name–affiliation database. Each time a paper is recognized as one of the “top five references” in another work, its authors and institutions receive credit. To keep the scoring fair, points are divided by the number of co-authors, ensuring balanced recognition across collaborations.

The result is a new kind of academic ranking: one that rewards universities not just for publishing often, but for producing research that endures, inspires, and drives the field forward. This approach highlights genuine scholarly influence and provides students, researchers, and institutions with a clearer picture of where the most impactful work is happening.

---

## Technical Overview

1. **PDF Downloader**  
   - Fetches research papers from top CS conferences.

2. **LLM Paper Scoring**  
   - Uses **deepseek-ai/DeepSeek-R1-Distill-Llama-8B** to extract the five most important references.  
   - Builds an impact score based on citation importance.

3. **Author Extraction**  
   - Matches papers to authors using the **DBLP API**  
   - Resolves naming inconsistencies and affiliation data 

---

## End User Documentation

### Explore the Rankings

**Visit the site:** *[Our Site](https://lianghui818.github.io/revolutionizing-higher-ed-rankings/public/)*  
Users can explore ranked university programs with filters and insights that go beyond traditional metrics.

---

### Features

- University rankings by research quality
- Continent-Based filters 
- Transparent scoring explanations 

---

### Troubleshooting

- **Empty results?** Reset filters or expand your criteria.
- **Slow load times?** CRefresh the page and check your connection.
- **Other issues?** Report via the GitHub Issues tab.

---

### FAQ

**Q: How are university rankings calculated?**  
A: Rankings are based on research quality, not just quantity. We use a language model to assess paper and return the 5 most important references to showcase impact.

**Q: Where does your data come from?**  
A: We use a curated dataset of CS papers from the top CS conferences and author metadata from DBLP, with PDF links for full-text analysis.

**Q: What makes this different from CS Rankings?**  
A: Unlike CS Rankings, we account for paper impact using LLM evaluation and fair contribution scoring via graph-based aggregation.

**Q: Why are some universities missing?**  
A: We only include institutions with verifiable publication data. Some universities may be excluded due to missing or ambiguous author affiliations.

**Q: How are co-authored papers scored?**  
A: Credit is distributed fairly among co-authors. In the influence graph, each citation link is weighted by 1 / number of authors.

**Q: Can I suggest improvements or report issues?**  
A: Yes! Please open an issue or pull request on our [GitHub repository](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings) to contribute or report bugs.

---

**All user and technical documentation is available in our GitHub repository README.**

---

## Project Resources

- [GitHub Repository](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings)
- [Developer Documentation Directory](https://github.com/Lianghui818/revolutionizing-higher-ed-rankings/tree/main/docs)

--- 
