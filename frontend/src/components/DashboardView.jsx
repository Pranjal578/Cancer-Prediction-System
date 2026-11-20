import React from 'react';
import { EXPERIMENT_METADATA, PRIORITY_RANKINGS, MODEL_DETAILS } from '../data/researchData';
import { Activity, Award, ShieldAlert, Zap, TrendingUp, CheckCircle, BarChart3, Clock, Sparkles } from 'lucide-react';

export default function DashboardView({ onNavigate }) {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Research Dashboard & Comparative Overview</h1>
        <p className="page-description">
          A scientific inquiry into prior-fitted tabular foundation networks (TabPFN-2.5) versus deep tabular transformers (FT-Transformer)
          and state-of-the-art gradient boosted ensembles (CatBoost, XGBoost) on the Wisconsin Diagnostic Breast Cancer (WDBC) benchmark.
        </p>
      </div>

      {/* Top Benchmark KPI Cards */}
      <div className="grid-4">
        <div className="kpi-card">
          <span className="kpi-label">Top Classification Accuracy</span>
          <span className="kpi-value" style={{ color: '#0284c7' }}>98.25%</span>
          <span className="kpi-sub">TabPFN-2.5 (Prior-Fitted)</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Top Discriminative ROC-AUC</span>
          <span className="kpi-value" style={{ color: '#059669' }}>0.9983</span>
          <span className="kpi-sub">CatBoost (Oblivious Trees)</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Best Probability Calibration</span>
          <span className="kpi-value" style={{ color: '#7c3aed' }}>0.0221</span>
          <span className="kpi-sub">Lowest ECE (TabPFN-2.5)</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Primary Candidate Specificity</span>
          <span className="kpi-value" style={{ color: '#d97706' }}>100.0%</span>
          <span className="kpi-sub">0 False Positives on Test Set</span>
        </div>
      </div>

      {/* Priority Rankings Leaderboard */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 className="card-title" style={{ margin: 0 }}>
            <Award size={22} color="#0284c7" />
            Official Research Leaderboard (Priority Order: Accuracy &gt; Precision &gt; F1 &gt; Recall)
          </h3>
          <button className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '6px 12px' }} onClick={() => onNavigate('COMPARISON')}>
            View All 12 Metrics
          </button>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Candidate Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>F1-Score</th>
                <th>Recall (Sensitivity)</th>
                <th>ROC-AUC</th>
                <th>Architectural Paradigm</th>
              </tr>
            </thead>
            <tbody>
              {PRIORITY_RANKINGS.map((row) => (
                <tr key={row.rank} style={{ backgroundColor: row.rank === 1 ? '#f0fdf4' : 'transparent' }}>
                  <td>
                    <span style={{ 
                      fontWeight: 800, 
                      color: row.rank === 1 ? '#15803d' : '#64748b',
                      fontSize: '1rem' 
                    }}>
                      #{row.rank}
                    </span>
                  </td>
                  <td>
                    <strong>{row.model}</strong>
                    {row.rank === 1 && (
                      <span className="best-pill" style={{ marginLeft: '8px' }}>PRIMARY WINNER</span>
                    )}
                  </td>
                  <td><strong>{row.accuracy}</strong></td>
                  <td>{row.precision}</td>
                  <td>{row.f1}</td>
                  <td>{row.recall}</td>
                  <td>{row.roc_auc}</td>
                  <td>
                    <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
                      {MODEL_DETAILS[row.model]?.type}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Benchmark Charts Grid */}
      <div className="grid-2">
        <div className="card">
          <h3 className="card-title">
            <BarChart3 size={20} color="#0284c7" />
            Held-Out Test Metric Comparison (All 4 Models)
          </h3>
          <img 
            src="/reports/metrics_comparison_bars.png" 
            alt="Metrics Comparison Bars" 
            className="report-image"
          />
          <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '10px' }}>
            Comparison across Accuracy, Precision, F1-Score, Recall, Sensitivity, Specificity, Balanced Accuracy, and MCC on untouched test samples (N=114).
          </p>
        </div>

        <div className="card">
          <h3 className="card-title">
            <TrendingUp size={20} color="#059669" />
            Receiver Operating Characteristic (ROC) Comparison
          </h3>
          <img 
            src="/reports/roc_curves_comparison.png" 
            alt="ROC Curves Comparison" 
            className="report-image"
          />
          <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '10px' }}>
            Multi-model ROC curves demonstrating near-perfect discriminative separation (AUC: 0.9769 to 0.9983) with CatBoost and TabPFN dominating the upper left quadrant.
          </p>
        </div>
      </div>

      {/* Model Profiles Grid */}
      <div className="card">
        <h3 className="card-title">
          <Zap size={20} color="#0284c7" />
          Candidate Architectures: At a Glance
        </h3>
        <div className="grid-4" style={{ marginBottom: 0 }}>
          {Object.entries(MODEL_DETAILS).map(([name, m]) => (
            <div key={name} style={{ border: '1px solid #e2e8f0', borderRadius: '8px', padding: '16px', background: '#f8fafc' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <strong style={{ fontSize: '0.95rem', color: '#0f172a' }}>{name}</strong>
                <span className="badge-unique" style={{ fontSize: '0.7rem' }}>UNIQUE</span>
              </div>
              <p style={{ fontSize: '0.78rem', color: '#64748b', marginBottom: '10px', height: '36px' }}>
                {m.paradigm}
              </p>
              <div style={{ fontSize: '0.78rem', color: '#334155', borderTop: '1px solid #e2e8f0', paddingTop: '8px' }}>
                <div>Train Time: <strong>{m.trainingTime}</strong></div>
                <div>Inference: <strong>{m.inferenceTime}</strong></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
