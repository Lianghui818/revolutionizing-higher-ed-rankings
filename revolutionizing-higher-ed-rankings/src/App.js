// App.js
import React from 'react';
import './App.css';
import Header from './components/Header';
import Main from './components/Main';
import Rankings from './components/Rankings';
import Footer from './components/Footer';

function App() {
  return (
    <div className="App">
      <Header />
      <Main />
      <Rankings />
      <Footer />
    </div>
  );
}

export default App;