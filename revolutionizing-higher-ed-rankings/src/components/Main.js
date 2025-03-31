// components/Main.js
import React from 'react';
import Section from './Section';

function Main() {
  const sections = [
    {
      id: "purpose",
      title: "Purpose of the Project",
      content: "The purpose of our project team is to deliver a website that fairly weighs all factors of a school's Computer Science department by calculating their rating with a methodology that takes a quality over quantity approach with valuing publications. This way, there will be a fair and publicly available site for prospective students to use when they want to find out which universities best fit their goals for a Computer Science degree. No matter what the student is looking for in a university, they will be able to find the information with Revolutionizing Higher Ed Rankings."
    },
    {
      id: "problem",
      title: "Problem Solved",
      content: "The problem that our project team faces is that there are many websites used for ranking universities, specifically in the realm of Computer Science, yet many of the options are not viable. Many sites are heavily biased towards 'lifetime achievements' or reputation for universities, while others make an effort to have objective and fair rankings. The best site that we have currently is CS Rankings, which counts publications that professors at universities have achieved and uses them as weights for a University's CS Ranking. The issue with that is that universities are now hiring professors as a result of their publication histories, with recent PhD candidates having many publications. This encourages universities to fund more research for more publications rather than focusing on doing in depth/meaningful or impactful work. There currently are not any viable options for countering this method."
    },
    {
      id: "vision",
      title: "Our Vision",
      content: "The vision for Revolutionizing Higher Ed Rankings is for any students hoping to pursue a degree in Computer Science to use our site for finding the right university for them. With a methodology that properly and fairly weighs all factors, there will be no stone unturned with what facets universities excel in. Students should eventually be able to search for schools based on specific focus areas like Artificial Intelligence, Graphic Design, or Web Development, as well as other factors like regional results. In the far future, there is a hope that hiring practices for universities will no longer be determined by the amount of publications a candidate has because of the impact of Revolutionizing Higher Ed Rankings."
    },
    {
      id: "site",
      title: "Our Site",
      content: "Please find our site",
      link: {
        text: "here",
        url: "https://web.engr.oregonstate.edu/~wangl9/revedrank/"
      }
    },
    {
      id: "help",
      title: "Documentation",
      content: "For detailed documentation, please visit our",
      link: {
        text: "documentation page",
        url: "https://github.com/Lianghui818/revolutionizing-higher-ed-rankings"
      }
    },
    {
      id: "repository",
      title: "Project Repository",
      content: "The source code and project files are available on our",
      link: {
        text: "GitHub repository",
        url: "https://github.com/Lianghui818/revolutionizing-higher-ed-rankings"
      }
    }
  ];

  return (
    <main>
      {sections.map((section) => (
        <Section 
          key={section.id}
          id={section.id}
          title={section.title}
          content={section.content}
          link={section.link}
        />
      ))}
      <hr className="divider" />
    </main>
  );
}

export default Main;