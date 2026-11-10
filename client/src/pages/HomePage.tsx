import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';
import ShaderCanvas from '../components/shared/ShaderCanvas';
import PromptBar from '../components/shared/PromptBar';
import './HomePage.css';

const HomePage: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (containerRef.current) {
      const elements = containerRef.current.querySelectorAll('.animate-item');
      gsap.fromTo(elements, 
        { y: 40, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, stagger: 0.1, ease: 'power2.out', delay: 0.2 }
      );
    }
  }, []);

  return (
    <div className="home-page" ref={containerRef}>
      <div className="shader-bg animate-fade-in-up">
        <ShaderCanvas />
      </div>
      
      <div className="home-content">
        <div className="greeting-section animate-item">
          <h1 className="font-display-lg greeting-title">
            What should we focus on?
          </h1>
        </div>
        
        <div className="prompt-container animate-item">
          <PromptBar variant="premium" />
        </div>
      </div>
    </div>
  );
};

export default HomePage;
