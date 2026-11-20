import React from 'react';
import { CALIBRATION_EVIDENCE } from '../data/researchData';
import { Target, ShieldAlert, CheckCircle2, TrendingUp, HelpCircle } from 'lucide-react';

export default function CalibrationView() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Predictive Reliability & Probability Calibration Analysis</h1>
        <p className="page-description">
          Assessing the alignment between predicted probabilities and empirical risk using reliability diagrams,
          Expected Calibration Error (ECE), Platt Scaling, and Isotonic Regression.
        </p>
      </div>

      {/* Clinical Rationale Alert */}
      <div className="card" style={{ borderLeft: '4px solid #7c3aed' }}>
        <h3 className="card-title">
          <HelpCircle size={22} color="#7c3aed" />
          Why Probability Calibration is Critical in Clinical Decision Support
        </h3>
        <p style={{ color: '#334155', fontSize: '0.92rem', lineHeight: '1.6' }}>
          In oncology decision support, a model that merely outputs discrete 0 or 1 labels is insufficient. Oncologists and patients require 
          <strong> calibrated risk estimates</strong>. If a model predicts an 80% probability of malignancy for a lesion, exactly 80 out of 100 such lesions must biologically prove malignant. Uncalibrated models (especially deep nets or unregularized boosting trees) often exhibit severe overconfidence, producing distorted risk assessments.
        </p>
      </div>

      {/* Visual Calibration Diagrams */}
      <div className="grid-2">
        <div className="card">
          <h3 className="card-title">
            <Target size={20} color="#0284c7" />
            Reliability Diagram: Pre vs. Post Calibration
          </h3>
          <img 
            src="/reports/calibration_comparison.png" 
            alt="Calibration Comparison Plot" 
            className="report-image"
          />
          <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '10px' }}>
            Diagonal dashed line represents perfect calibration (y = x). Curves closer to the diagonal represent reliable risk probabilities.
          </p>
        </div>

        <div className="card">
          <h3 className="card-title">
            <TrendingUp size={20} color="#059669" />
            Calibration Techniques Investigated
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '0.88rem', color: '#334155' }}>
            <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <strong style={{ color: '#0f172a' }}>1. Raw Probabilities:</strong>
              <p style={{ color: '#64748b', marginTop: '4px' }}>
                Unmodified softmax / tree margin probabilities generated directly by the candidate architecture.
              </p>
            </div>

            <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <strong style={{ color: '#0f172a' }}>2. Platt Scaling (Logistic Sigmoid):</strong>
              <p style={{ color: '#64748b', marginTop: '4px' }}>
                Fits a univariate logistic regression P(y=1|f) = 1 / (1 + exp(A * f + B)) strictly on out-of-fold cross-validation predictions.
              </p>
            </div>

            <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <strong style={{ color: '#0f172a' }}>3. Isotonic Regression (Non-Parametric):</strong>
              <p style={{ color: '#64748b', marginTop: '4px' }}>
                Fits a non-decreasing step function using the pair-adjacent violators (PAV) algorithm, eliminating parametric assumptions.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Calibration Comparison Table */}
      <div className="card">
        <h3 className="card-title">
          <CheckCircle2 size={20} color="#0284c7" />
          Expected Calibration Error (ECE) & Brier Score Matrix
        </h3>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Raw ECE (10 Bins)</th>
                <th>Raw Brier Score</th>
                <th>Platt Scaled ECE</th>
                <th>Platt Brier Score</th>
                <th>Isotonic Scaled ECE</th>
                <th>Isotonic Brier Score</th>
                <th>Calibration Quality</th>
              </tr>
            </thead>
            <tbody>
              {CALIBRATION_EVIDENCE.map((row) => (
                <tr key={row.model} style={{ backgroundColor: row.model.includes('TabPFN') ? '#f0fdf4' : 'transparent' }}>
                  <td><strong>{row.model}</strong></td>
                  <td><strong style={{ color: '#0284c7' }}>{row.rawEce.toFixed(4)}</strong></td>
                  <td>{row.rawBrier.toFixed(4)}</td>
                  <td>{row.plattEce.toFixed(4)}</td>
                  <td>{row.plattBrier.toFixed(4)}</td>
                  <td>{row.isoEce.toFixed(4)}</td>
                  <td>{row.isoBrier.toFixed(4)}</td>
                  <td>
                    <span className="best-pill">
                      {row.model.includes('TabPFN') ? "SUPERIOR IN-CONTEXT CALIBRATION" : "RELIABLE POST-SCALING"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: '16px', background: '#f8fafc', padding: '14px', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.85rem', color: '#475569' }}>
          <strong>Key Empirical Finding:</strong> TabPFN-2.5 achieves the lowest raw ECE (<strong>0.0221</strong>) and lowest raw Brier score (<strong>0.0164</strong>) without requiring post-hoc calibration, because its prior-fitted Bayesian architecture naturally yields well-calibrated posterior predictive distributions.
        </div>
      </div>
    </div>
  );
}
