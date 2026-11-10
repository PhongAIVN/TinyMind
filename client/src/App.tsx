import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import ChatPage from './pages/ChatPage';
import HistoryPage from './pages/HistoryPage';
import ExplorePage from './pages/ExplorePage';
import SettingsPage from './pages/SettingsPage';
import SideNavRail from './components/layout/SideNavRail';
import TopAppBar from './components/layout/TopAppBar';
import { ChatSettingsProvider } from './state/ChatSettings';
import './index.css';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <ChatSettingsProvider>
        <div className="app-container">
          <SideNavRail />
          <TopAppBar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/chat/:id?" element={<ChatPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/explore" element={<ExplorePage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </ChatSettingsProvider>
    </BrowserRouter>
  );
}

export default App;
