import React from 'react';
import { ARCHITECTURAL_TAXONOMY } from '../data/researchData';
import { Layers, Cpu, GitMerge, Check, AlertCircle, FileText, ArrowRight, Zap } from 'lucide-react';

export default function ModelParadigmProofView() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Architectural Proof: Unique Model vs. Hybrid Model Analysis</h1>
        <p className="page-description">
          Formal mathematical and structural categorization establishing whether each candidate architecture in this research
          constitutes a <strong>Unique Model (Single-Paradigm)</strong> or a <strong>Hybrid Model (Heterogeneous Multi-Paradigm Ensemble)</strong>.
        </p>
      </div>

      {/* Conceptual Definitions Card */}
      <div className="grid-2">
        <div className="card" style={{ borderTop: '4px solid #0284c7' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <Cpu size={22} color="#0284c7" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Definition: Unique Model (Single-Paradigm)</h3>
          </div>
          <p style={{ fontSize: '0.9rem', color: '#475569', lineHeight: '1.6' }}>
            {ARCHITECTURAL_TAXONOMY.definitions.uniqueModel}
          </p>
          <div style={{ marginTop: '16px', background: '#f0f9ff', padding: '12px', borderRadius: '6px', fontSize: '0.82rem', color: '#0369a1' }}>
            <strong>Hallmarks:</strong> Single loss gradient formulation, unified representation space, homogeneous parameter updates, self-contained inductive bias.
          </div>
        </div>

        <div className="card" style={{ borderTop: '4px solid #d97706' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <GitMerge size={22} color="#d97706" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Definition: Hybrid Model (Multi-Paradigm)</h3>
          </div>
          <p style={{ fontSize: '0.9rem', color: '#475569', lineHeight: '1.6' }}>
            {ARCHITECTURAL_TAXONOMY.definitions.hybridModel}
          </p>
          <div style={{ marginTop: '16px', background: '#fef3c7', padding: '12px', borderRadius: '6px', fontSize: '0.82rem', color: '#92400e' }}>
            <strong>Hallmarks:</strong> Multi-stage inference, disparate base algorithm families (e.g. attention + trees + neural heads), meta-learner loss arbitration.
          </div>
        </div>
      </div>

      {/* Proof Breakdown for Each Architecture */}
      <div className="card">
        <h3 className="card-title">
          <Layers size={22} color="#0284c7" />
          Rigorous Classification & Mathematical Proof
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {ARCHITECTURAL_TAXONOMY.models.map((item, idx) => (
            <div key={idx} style={{ 
              border: '1px solid #e2e8f0', 
              borderRadius: '10px', 
              padding: '20px',
              backgroundColor: item.isHybrid ? '#fffbeb' : '#ffffff'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                <div>
                  <h4 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a' }}>{item.name}</h4>
                  <p style={{ fontSize: '0.85rem', color: '#64748b' }}>{item.category}</p>
                </div>

                <span className={item.isHybrid ? 'badge-hybrid' : 'badge-unique'}>
                  {item.classification}
                </span>
              </div>

              <div style={{ marginBottom: '14px', fontSize: '0.92rem', color: '#334155', lineHeight: '1.6' }}>
                <strong style={{ color: '#0f172a' }}>Mathematical Proof & Grounding:</strong> {item.proofOfClassification}
              </div>

              <div style={{ marginBottom: '16px' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Core Mathematical Formulation:
                </span>
                <div className="code-block" style={{ marginTop: '4px' }}>
                  {item.mathematicalFormula}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Structural Verification Components:
                </span>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '8px', marginTop: '8px' }}>
                  {item.structuralComponents.map((comp, cIdx) => (
                    <div key={cIdx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.82rem', color: '#475569' }}>
                      <Check size={16} color={item.isHybrid ? '#d97706' : '#0284c7'} />
                      <span>{comp}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Synthesis Taxonomy Matrix */}
      <div className="card">
        <h3 className="card-title">
          <FileText size={22} color="#0284c7" />
          Architectural Paradigm Taxonomy Matrix
        </h3>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Classification</th>
                <th>Underlying Representation</th>
                <th>Training Mechanism</th>
                <th>Inference Mechanism</th>
                <th>Hyperparameter Tuning Sensitivity</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>TabPFN-2.5</strong></td>
                <td><span className="badge-unique">UNIQUE MODEL</span></td>
                <td>Prior-fitted attention embeddings</td>
                <td>Offline pre-training (synthetic DAGs)</td>
                <td>Single forward pass (No SGD)</td>
                <td>Zero tuning needed (Deterministic)</td>
              </tr>
              <tr>
                <td><strong>FT-Transformer</strong></td>
                <td><span className="badge-unique">UNIQUE MODEL</span></td>
                <td>Feature Tokenizer + Pre-LN MHSA</td>
                <td>AdamW backpropagation</td>
                <td>Feed-forward token pooling</td>
                <td>High (learning rate, depth, heads)</td>
              </tr>
              <tr>
                <td><strong>CatBoost</strong></td>
                <td><span className="badge-unique">UNIQUE MODEL</span></td>
                <td>Oblivious symmetric decision trees</td>
                <td>Ordered gradient boosting</td>
                <td>Binary bitmask index lookup</td>
                <td>Medium (depth, l2_leaf_reg)</td>
              </tr>
              <tr>
                <td><strong>XGBoost</strong></td>
                <td><span className="badge-unique">UNIQUE MODEL</span></td>
                <td>Greedy split-finding decision trees</td>
                <td>2nd-order Taylor approximation</td>
                <td>Tree path traversal</td>
                <td>High (max_depth, eta, gamma)</td>
              </tr>
              <tr style={{ backgroundColor: '#fffbeb' }}>
                <td><strong>Stacking / Soft Voting</strong></td>
                <td><span className="badge-hybrid">HYBRID MODEL</span></td>
                <td>Heterogeneous probability fusion</td>
                <td>OOF Cross-Validation meta-model</td>
                <td>Weighted meta-risk arbitration</td>
                <td>Compound (Base models + Meta-learner)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: '16px', background: '#f8fafc', padding: '14px', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.85rem', color: '#475569' }}>
          <strong>Research Conclusion on Uniqueness:</strong> TabPFN-2.5 is a completely <em>unique tabular foundation model</em> that breaks from classical empirical risk minimization by performing inference strictly in-context. When combining TabPFN with CatBoost and FT-Transformer via Stacking, the resulting system operates as a <em>true hybrid multi-paradigm architecture</em>.
        </div>
      </div>
    </div>
  );
}
