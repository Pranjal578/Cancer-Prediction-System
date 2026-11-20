import React from 'react';
import { EXPERIMENT_METADATA } from '../data/researchData';
import { Database, FileSpreadsheet, Info, AlertTriangle, Layers, PieChart, Activity } from 'lucide-react';

export default function DatasetExplorerView() {
  const featureCategories = [
    { name: "Radius", desc: "Mean of distances from center to points on the cell nucleus perimeter." },
    { name: "Texture", desc: "Standard deviation of gray-scale intensity values across nuclear chromatin." },
    { name: "Perimeter", desc: "Total length of the surrounding boundary of the cell nucleus." },
    { name: "Area", desc: "Cross-sectional surface area bounded within nuclear contour." },
    { name: "Smoothness", desc: "Local variation in radius lengths, measuring contour irregularity." },
    { name: "Compactness", desc: "Calculated as (perimeter^2 / area - 1.0), measuring cell density." },
    { name: "Concavity", desc: "Severity of concave indentations along the nuclear perimeter." },
    { name: "Concave Points", desc: "Count of localized concave indentations on the boundary." },
    { name: "Symmetry", desc: "Nuclear bilateral reflection symmetry relative to perpendicular axis." },
    { name: "Fractal Dimension", desc: "Boundary complexity calculated using the 'coastline approximation' (Mandelbrot)." }
  ];

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Dataset Exploration: Wisconsin Diagnostic Breast Cancer (WDBC)</h1>
        <p className="page-description">
          Ground-truth digitized morphometric measurements extracted from Fine Needle Aspirates (FNA) of human breast masses.
          Curated by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin).
        </p>
      </div>

      {/* Cohort Composition KPIs */}
      <div className="grid-4">
        <div className="kpi-card">
          <span className="kpi-label">Total Biopsy Cohort</span>
          <span className="kpi-value">569</span>
          <span className="kpi-sub">Patient FNA aspirates</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Benign (0) Cases</span>
          <span className="kpi-value" style={{ color: '#059669' }}>357</span>
          <span className="kpi-sub">62.74% of cohort (Class 0)</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Malignant (1) Cases</span>
          <span className="kpi-value" style={{ color: '#dc2626' }}>212</span>
          <span className="kpi-sub">37.26% of cohort (Positive Class)</span>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Continuous Features</span>
          <span className="kpi-value">30</span>
          <span className="kpi-sub">10 Mean + 10 SE + 10 Worst</span>
        </div>
      </div>

      {/* Cohort Split & Methodology Card */}
      <div className="grid-2">
        <div className="card">
          <h3 className="card-title">
            <PieChart size={20} color="#0284c7" />
            Class Distribution & Stratification
          </h3>
          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.88rem' }}>
              <span><strong>Benign Cases:</strong> 357 (62.7%)</span>
              <span style={{ color: '#059669', fontWeight: 600 }}>Non-Malignant Histology</span>
            </div>
            <div className="prob-bar-bg" style={{ height: '14px' }}>
              <div className="prob-bar-fill benign" style={{ width: '62.74%' }}></div>
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.88rem' }}>
              <span><strong>Malignant Cases:</strong> 212 (37.3%)</span>
              <span style={{ color: '#dc2626', fontWeight: 600 }}>Confirmed Carcinoma</span>
            </div>
            <div className="prob-bar-bg" style={{ height: '14px' }}>
              <div className="prob-bar-fill malignant" style={{ width: '37.26%' }}></div>
            </div>
          </div>

          <div style={{ fontSize: '0.85rem', color: '#475569', background: '#f8fafc', padding: '12px', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
            <strong>Leakage Prevention:</strong> Partitioned using stratified sampling into 80% development (455 samples) and 20% untouched held-out testing (114 samples). Preprocessing statistics (imputation medians, standard scaling) are computed exclusively on training folds.
          </div>
        </div>

        <div className="card">
          <h3 className="card-title">
            <Layers size={20} color="#0284c7" />
            Feature Dimensionality Structure
          </h3>
          <p style={{ fontSize: '0.9rem', color: '#475569', marginBottom: '14px' }}>
            Each patient biopsy image yields 10 fundamental nuclear characteristics computed across cell nuclei boundaries, recorded in 3 statistical tiers:
          </p>

          <ul style={{ listStyle: 'none', fontSize: '0.85rem', color: '#334155', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <li style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>1. Mean Tier (Features 1–10):</strong> Average measurement across all segmented cell nuclei in the patient's FNA biopsy sample.
            </li>
            <li style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>2. Standard Error Tier (Features 11–20):</strong> Standard error of the mean, quantifying intra-specimen cell nuclear heterogeneity.
            </li>
            <li style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>3. Worst Tier (Features 21–30):</strong> Mean of the three largest (most extreme) values, capturing peak morphological atypia.
            </li>
          </ul>
        </div>
      </div>

      {/* Feature Catalog Table */}
      <div className="card">
        <h3 className="card-title">
          <FileSpreadsheet size={20} color="#0284c7" />
          The 10 Base Nuclear Morphometric Characteristics
        </h3>
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Feature Name</th>
                <th>Biological & Morphological Description</th>
                <th>Clinical Pathology Relevance</th>
              </tr>
            </thead>
            <tbody>
              {featureCategories.map((item, idx) => (
                <tr key={idx}>
                  <td><strong>{item.name}</strong></td>
                  <td>{item.desc}</td>
                  <td>
                    <span style={{ fontSize: '0.82rem', color: '#475569' }}>
                      Elevated values strongly correlate with nuclear pleomorphism, rapid mitosis, and malignant dysplasia.
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
