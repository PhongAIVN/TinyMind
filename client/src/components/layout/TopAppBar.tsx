import React from 'react';
// import { GITHUB_URL, HUGGINGFACE_URL } from '../../config/app_config';
import './TopAppBar.css';

const TopAppBar: React.FC = () => {
  return (
    <header className="top-app-bar">
      <nav className="top-nav-links" aria-label="Liên kết dự án">
        {/* <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="font-label-md">GitHub</a> */}
        {/* <a href={HUGGINGFACE_URL} target="_blank" rel="noopener noreferrer" className="font-label-md">Hugging Face</a> */}
      </nav>
    </header>
  );
};

export default TopAppBar;
