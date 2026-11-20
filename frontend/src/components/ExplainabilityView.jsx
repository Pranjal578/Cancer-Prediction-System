import React, { useState } from 'react';
import { TOP_10_FEATURES } from '../data/researchData';
import { Eye, HelpCircle, Layers, CheckCircle, AlertTriangle, BookOpen } from 'lucide-react';

export default function ExplainabilityView() {
  const [selectedModel, setSelectedModel] = useState('tabpfn');

  const modelOptions = [
    { key: 'tabpfn', label: 'TabPFN-2.5', explainer: 'KernelExplainer (Subsampled background)' },
    { key: 'catboost', label: 'CatBoost', explainer: 'TreeExplainer (Exact polynomial tree SHAP)' },
    { key: 'xgboost', label: 'XGBoost', explainer: 'TreeExplainer (Exact polynomial tree SHAP)' },
    { key: 'ft_transformer', label: 'FT-Transformer', explainer: 'KernelExplainer (Standardized background)' },
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Explainable Artificial Intelligence (XAI): SHAP Studio</h1>
        <p className="page-description">
          Grounding black-box predictive models in cooperative game theory using <strong>SHapley Additive exPlanations (SHAP)</strong>
          to quantify feature attributions at both global cohort and local patient levels.
        </p>
      </div>

      {/* Model Selector Bar */}
      <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        <strong style={{ fontSize: '0.95rem' }}>Select Model Architecture:</strong>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {modelOptions.map(m => (
            <button
              key={m.key}
              className={`btn ${selectedModel === m.key ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setSelectedModel(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <span style={{ marginLeft: 'auto', fontSize: '0.8rem', color: '#64748b' }}>
          Explainer: <strong>{modelOptions.find(o => o.key === selectedModel)?.explainer}</strong>
        </span>
      </div>

      {/* SHAP Visualizations */}
      <div className="grid-2">
        <div className="card">
          <h3 className="card-title">
            <Eye size={20} color="#0284c7" />
            SHAP Beeswarm Plot: {modelOptions.find(o => o.key === selectedModel)?.label}
          </h3>
          <img 
            src={`/reports/${selectedModel}_shap_beeswarm.png`} 
            alt={`${selectedModel} SHAP Beeswarm`} 
            className="report-image"
          />
          <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '10px', lineHeight: '1.5' }}>
            Each dot represents a patient biopsy. Color encodes feature value (<strong>Red = High</strong>, <strong>Blue = Low</strong>).
            Horizontal position indicates whether that feature value increases (positive SHAP &gt; 0) or decreases (negative SHAP &lt; 0) the likelihood of malignancy.
          </p>
        </div>

        <div className="card">
          <h3 className="card-title">
            <Layers size={20} color="#0284c7" />
            Global Mean |SHAP| Attribution Bar Chart
          </h3>
          <img 
            src={`/reports/${selectedModel}_shap_bar.png`} 
            alt={`${selectedModel} SHAP Bar`} 
            className="report-image"
          />
          <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '10px', lineHeight: '1.5' }}>
            Ranks features strictly by mean absolute Shapley value across the cohort. Higher values reflect higher overall contribution to the model's discriminative decisions.
          </p>
        </div>
      </div>

      {/* Top 10 Biological Tumour Characteristics */}
      <div className="card">
        <h3 className="card-title">
          <BookOpen size={20} color="#0284c7" />
          Top 10 Most Influential Tumour Characteristics Across Models
        </h3>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Diagnostic Feature</th>
                <th>Mean |SHAP|</th>
                <th>Biological & Morphological Context in Pathology</th>
              </tr>
            </thead>
            <tbody>
              {TOP_10_FEATURES.map((item) => (
                <tr key={item.rank}>
                  <td><strong>#{item.rank}</strong></td>
                  <td><code style={{ color: '#0284c7', fontWeight: 600 }}>{item.feature}</code></td>
                  <td><strong>{item.meanShap.toFixed(3)}</strong></td>
                  <td style={{ fontSize: '0.85rem', color: '#475569' }}>{item.biologicalContext}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: '16px', background: '#fef3c7', padding: '14px', borderRadius: '8px', border: '1px solid #fde68a', fontSize: '0.85rem', color: '#92400e' }}>
          <strong>⚠️ Association vs. Causation Caution:</strong> Statistical feature importance identifies morphological measurements strongly correlated with malignancy (e.g., enlarged nuclear perimeter, high concave points). While these features reflect tumor atypia, they represent diagnostic correlations rather than direct biological etiologies.
        </div>
      </div>

      {/* Mathematical Formulation */}
      <div className="card">
        <h3 className="card-title">
          <HelpCircle size={20} color="#0284c7" />
          Mathematical Foundation: Cooperative Game Theory
        </h3>
        <p style={{ fontSize: '0.9rem', color: '#334155', marginBottom: '12px' }}>
          The Shapley value (phi_i) allocates payout to feature i fairly according to its marginal contribution across all possible feature subsets S:
        </p>
        <div className="code-block">
          {"\u03C6_i(v) = \u2211_{S \u2286 N \\ {i}} [ |S|!(|N| - |S| - 1)! / |N|! ] \u00D7 [ v(S \u222A {i}) - v(S) ]"}
        </div>
        <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '10px' }}>
          Where $N$ is the set of all 30 FNA features, $S$ is a coalition excluding feature $i$, and $v(S)$ is the conditional expectation of the model prediction.
        </p>
      </div>
    </div>
  );
}
