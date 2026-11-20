import React, { useState } from 'react';
import { COMPARISON_TABLE, PRIORITY_RANKINGS, MODEL_DETAILS } from '../data/researchData';
import { BarChart3, Award, TrendingUp, Grid, CheckCircle2, ChevronRight, Activity } from 'lucide-react';

export default function ComparisonView() {
  const [activeTab, setActiveTab] = useState('TABLE');

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Comprehensive 12-Metric Comparative Evaluation</h1>
        <p className="page-description">
          Evaluation of all four models on the untouched held-out test partition (N=114) across 12 statistical metrics,
          prioritized according to clinical diagnostic importance: Accuracy &gt; Precision &gt; F1-Score &gt; Recall.
        </p>
      </div>

      {/* Tabs */}
      <div className="tabs">
        <div 
          className={`tab ${activeTab === 'TABLE' ? 'active' : ''}`}
          onClick={() => setActiveTab('TABLE')}
        >
          Consolidated 12-Metric Matrix
        </div>
        <div 
          className={`tab ${activeTab === 'CHARTS' ? 'active' : ''}`}
          onClick={() => setActiveTab('CHARTS')}
        >
          ROC, PR & Metric Bar Charts
        </div>
        <div 
          className={`tab ${activeTab === 'CONFUSION' ? 'active' : ''}`}
          onClick={() => setActiveTab('CONFUSION')}
        >
          Confusion Matrix Grid
        </div>
      </div>

      {activeTab === 'TABLE' && (
        <>
          <div className="card">
            <h3 className="card-title">
              <Award size={22} color="#0284c7" />
              Benchmark Results Across All 12 Metrics
            </h3>

            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Evaluation Metric</th>
                    <th>TabPFN-2.5 (Candidate)</th>
                    <th>CatBoost (Benchmark)</th>
                    <th>FT-Transformer (Deep Learning)</th>
                    <th>XGBoost (Benchmark)</th>
                    <th>Best Performer</th>
                  </tr>
                </thead>
                <tbody>
                  {COMPARISON_TABLE.map((row, idx) => (
                    <tr key={idx} style={{ backgroundColor: row.best.includes('TabPFN') ? '#f8fafc' : 'transparent' }}>
                      <td><strong>{row.metric}</strong></td>
                      <td>
                        <span style={{ 
                          fontWeight: row.best.includes('TabPFN') ? 800 : 500,
                          color: row.best.includes('TabPFN') ? '#0284c7' : 'inherit'
                        }}>
                          {row.format === 'pct' ? `${(row.tabpfn * 100).toFixed(2)}%` : row.tabpfn.toFixed(4)}
                        </span>
                      </td>
                      <td>
                        <span style={{ 
                          fontWeight: row.best.includes('CatBoost') ? 800 : 500,
                          color: row.best.includes('CatBoost') ? '#059669' : 'inherit'
                        }}>
                          {row.format === 'pct' ? `${(row.catboost * 100).toFixed(2)}%` : row.catboost.toFixed(4)}
                        </span>
                      </td>
                      <td>
                        <span style={{ 
                          fontWeight: row.best.includes('FT-Transformer') ? 800 : 500,
                          color: row.best.includes('FT-Transformer') ? '#7c3aed' : 'inherit'
                        }}>
                          {row.format === 'pct' ? `${(row.ft_transformer * 100).toFixed(2)}%` : row.ft_transformer.toFixed(4)}
                        </span>
                      </td>
                      <td>
                        <span style={{ 
                          fontWeight: row.best.includes('XGBoost') ? 800 : 500,
                          color: row.best.includes('XGBoost') ? '#d97706' : 'inherit'
                        }}>
                          {row.format === 'pct' ? `${(row.xgboost * 100).toFixed(2)}%` : row.xgboost.toFixed(4)}
                        </span>
                      </td>
                      <td>
                        <span className="best-pill">
                          {row.best}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '16px', fontSize: '0.82rem', color: '#64748b', display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
              <span>* Positive Class = Malignant (1)</span>
              <span>* Brier Score & Log Loss: Lower values indicate superior probabilistic calibration</span>
              <span>* Expected Calibration Error (ECE): 10 equal-width probability bins</span>
            </div>
          </div>

          <div className="card">
            <h3 className="card-title">
              <CheckCircle2 size={20} color="#059669" />
              Scientific Performance Synthesis
            </h3>
            <p style={{ fontSize: '0.92rem', color: '#334155', lineHeight: '1.6' }}>
              <strong>TabPFN-2.5</strong> achieved the highest classification accuracy (<strong>98.25%</strong>), highest F1-score (<strong>0.9756</strong>), and highest sensitivity/recall (<strong>95.24%</strong>), with zero false positives (<strong>100% Precision</strong> and <strong>100% Specificity</strong>). It simultaneously delivered the lowest Expected Calibration Error (<strong>0.0221</strong>) and lowest Brier score (<strong>0.0164</strong>) without requiring hyperparameter tuning, validating the superior empirical efficacy of in-context tabular foundation transformers.
            </p>
          </div>
        </>
      )}

      {activeTab === 'CHARTS' && (
        <div className="grid-2">
          <div className="card">
            <h3 className="card-title">
              <BarChart3 size={20} color="#0284c7" />
              12-Metric Comparative Bar Chart
            </h3>
            <img src="/reports/metrics_comparison_bars.png" alt="Metrics Comparison" className="report-image" />
          </div>

          <div className="card">
            <h3 className="card-title">
              <TrendingUp size={20} color="#059669" />
              Receiver Operating Characteristic (ROC) Curves
            </h3>
            <img src="/reports/roc_curves_comparison.png" alt="ROC Curves" className="report-image" />
          </div>

          <div className="card">
            <h3 className="card-title">
              <Activity size={20} color="#7c3aed" />
              Precision-Recall (PR) Curves
            </h3>
            <img src="/reports/pr_curves_comparison.png" alt="Precision-Recall Curves" className="report-image" />
          </div>

          <div className="card">
            <h3 className="card-title">
              <Grid size={20} color="#d97706" />
              Probability Calibration Reliability Comparison
            </h3>
            <img src="/reports/calibration_comparison.png" alt="Calibration Reliability" className="report-image" />
          </div>
        </div>
      )}

      {activeTab === 'CONFUSION' && (
        <div className="card">
          <h3 className="card-title">
            <Grid size={20} color="#0284c7" />
            Held-Out Test Confusion Matrices (2x2 Multi-Model Grid)
          </h3>
          <img src="/reports/confusion_matrices_grid.png" alt="Confusion Matrices Grid" className="report-image" />

          <div className="grid-4" style={{ marginTop: '20px' }}>
            {Object.entries(MODEL_DETAILS).map(([name, m]) => (
              <div key={name} style={{ border: '1px solid #e2e8f0', borderRadius: '8px', padding: '14px', background: '#f8fafc' }}>
                <strong style={{ fontSize: '0.9rem', color: '#0f172a' }}>{name}</strong>
                <div style={{ fontSize: '0.8rem', color: '#475569', marginTop: '6px' }}>
                  <div>True Negatives (TN): <strong>{m.confusionMatrix.tn}</strong></div>
                  <div>False Positives (FP): <strong style={{ color: m.confusionMatrix.fp > 0 ? '#dc2626' : '#059669' }}>{m.confusionMatrix.fp}</strong></div>
                  <div>False Negatives (FN): <strong style={{ color: '#dc2626' }}>{m.confusionMatrix.fn}</strong></div>
                  <div>True Positives (TP): <strong>{m.confusionMatrix.tp}</strong></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
