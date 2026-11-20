import React, { useState } from 'react';
import { 
  LayoutDashboard, 
  Database, 
  BarChart2, 
  Activity, 
  Eye, 
  Target, 
  BookOpen, 
  Cpu, 
  FileText, 
  AlertTriangle,
  FlaskConical,
  ShieldCheck,
  ChevronRight
} from 'lucide-react';

import DashboardView from './components/DashboardView';
import DatasetExplorerView from './components/DatasetExplorerView';
import ComparisonView from './components/ComparisonView';
import PredictionWorkbenchView from './components/PredictionWorkbenchView';
import ExplainabilityView from './components/ExplainabilityView';
import CalibrationView from './components/CalibrationView';
import SciIeeeProofView from './components/SciIeeeProofView';
import ModelParadigmProofView from './components/ModelParadigmProofView';
import ResearchReportView from './components/ResearchReportView';

export default function App() {
  const [currentView, setCurrentView] = useState('DASHBOARD');

  const navItems = [
    { key: 'DASHBOARD', label: '1. Research Dashboard', icon: LayoutDashboard },
    { key: 'DATASET', label: '2. Dataset Exploration', icon: Database },
    { key: 'COMPARISON', label: '3. 12-Metric Benchmark', icon: BarChart2 },
    { key: 'PREDICTION', label: '4. Tumour Prediction', icon: Activity },
    { key: 'EXPLAINABILITY', label: '5. Explainable AI (SHAP)', icon: Eye },
    { key: 'CALIBRATION', label: '6. Probability Calibration', icon: Target },
    { key: 'SCI_IEEE_PROOF', label: '7. SCI & IEEE Proof', icon: BookOpen, badge: 'VERIFIED' },
    { key: 'PARADIGM_PROOF', label: '8. Unique vs Hybrid Proof', icon: Cpu, badge: 'PROOF' },
    { key: 'REPORT', label: '9. IEEE Research Paper', icon: FileText },
  ];

  const renderActiveView = () => {
    switch (currentView) {
      case 'DASHBOARD':
        return <DashboardView onNavigate={(view) => setCurrentView(view)} />;
      case 'DATASET':
        return <DatasetExplorerView />;
      case 'COMPARISON':
        return <ComparisonView />;
      case 'PREDICTION':
        return <PredictionWorkbenchView />;
      case 'EXPLAINABILITY':
        return <ExplainabilityView />;
      case 'CALIBRATION':
        return <CalibrationView />;
      case 'SCI_IEEE_PROOF':
        return <SciIeeeProofView />;
      case 'PARADIGM_PROOF':
        return <ModelParadigmProofView />;
      case 'REPORT':
        return <ResearchReportView />;
      default:
        return <DashboardView onNavigate={(view) => setCurrentView(view)} />;
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-title">
            <FlaskConical size={24} color="#38bdf8" />
            <span>Breast Cancer XAI</span>
          </div>
          <div className="sidebar-subtitle">
            Comparative Research Framework
          </div>
          <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '6px' }}>
            TabPFN-2.5 &bull; CatBoost &bull; FT-Transformer &bull; XGBoost
          </div>
        </div>

        <ul className="nav-list">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentView === item.key;
            return (
              <li
                key={item.key}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => setCurrentView(item.key)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
                {item.badge && (
                  <span className="nav-badge-pill">
                    {item.badge}
                  </span>
                )}
              </li>
            );
          })}
        </ul>

        <div className="sidebar-footer">
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <ShieldCheck size={16} color="#10b981" />
            <strong style={{ color: '#cbd5e1' }}>Academic Research System</strong>
          </div>
          <div>569 Samples &bull; 30 FNA Features &bull; 12 Metrics</div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Safety & Educational Disclaimer */}
        <div className="disclaimer-banner">
          <AlertTriangle size={20} color="#f59e0b" style={{ flexShrink: 0 }} />
          <div>
            <strong>ACADEMIC RESEARCH & BENCHMARKING SYSTEM ONLY:</strong>
            {" "}This application demonstrates empirical machine learning and explainable AI benchmarks on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
            Inferences are statistical models for educational exploration and MUST NOT substitute for biopsy histology, oncology evaluation, or physician diagnosis.
          </div>
        </div>

        {/* Dynamic View Component */}
        {renderActiveView()}
      </main>
    </div>
  );
}
