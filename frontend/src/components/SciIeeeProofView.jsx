import React, { useState } from 'react';
import { SCI_IEEE_EVIDENCE } from '../data/researchData';
import { BookOpen, ExternalLink, Award, CheckCircle2, Search, Filter, ShieldCheck, Database } from 'lucide-react';

export default function SciIeeeProofView() {
  const [filter, setFilter] = useState('ALL');
  const [search, setSearch] = useState('');

  const filteredPapers = SCI_IEEE_EVIDENCE.filter(paper => {
    const matchesFilter = filter === 'ALL' || 
      (filter === 'SCI' && paper.indexing.includes('SCI')) ||
      (filter === 'IEEE' && paper.indexing.includes('IEEE'));
    const matchesSearch = paper.title.toLowerCase().includes(search.toLowerCase()) ||
      paper.authors.toLowerCase().includes(search.toLowerCase()) ||
      paper.venue.toLowerCase().includes(search.toLowerCase()) ||
      paper.datasetProof.toLowerCase().includes(search.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Academic Literature Proof: SCI & IEEE Verification</h1>
        <p className="page-description">
          Rigorous bibliographic evidence and digital object identifiers (DOIs) proving the Breast Cancer Wisconsin Diagnostic (WDBC) dataset
          is the recognized gold-standard benchmark in Science Citation Index (SCI) and IEEE Xplore peer-reviewed literature.
        </p>
      </div>

      {/* Proof Summary Card */}
      <div className="card" style={{ borderLeft: '4px solid #0284c7' }}>
        <h3 className="card-title">
          <ShieldCheck size={22} color="#0284c7" />
          Evidence Summary: WDBC in Peer-Reviewed Research
        </h3>
        <p style={{ color: '#475569', fontSize: '0.95rem', marginBottom: '16px' }}>
          The Wisconsin Diagnostic Breast Cancer (WDBC) dataset was created by researchers at the <strong>University of Wisconsin Clinical Sciences Center</strong> (Dr. William H. Wolberg, Dr. W. Nick Street, and Dr. Olvi L. Mangasarian) and introduced in seminal <strong>SCI (Science Citation Index)</strong> and <strong>IEEE</strong> papers. It is deposited in the <strong>UCI Machine Learning Repository</strong> and has garnered over <strong>15,000+ total scientific citations</strong> across oncology, biomedical engineering, and artificial intelligence journals.
        </p>

        <div className="grid-3" style={{ marginBottom: '0' }}>
          <div style={{ background: '#f8fafc', padding: '14px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <Award size={18} color="#0284c7" />
              <strong style={{ fontSize: '0.9rem' }}>SCI / SCIE Journals</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: '#64748b' }}>
              Published in high-impact Q1 oncology & AI journals: <em>Cancer Letters (IF: 9.7)</em>, <em>Expert Systems with Applications (IF: 8.5)</em>, <em>Operations Research</em>, and <em>Computers in Biology and Medicine (IF: 7.7)</em>.
            </p>
          </div>

          <div style={{ background: '#f8fafc', padding: '14px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <Database size={18} color="#059669" />
              <strong style={{ fontSize: '0.9rem' }}>IEEE Xplore Venues</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: '#64748b' }}>
              Extensively indexed across <em>IEEE Transactions on Biomedical Engineering</em>, <em>IEEE Access</em>, and IEEE flagship conferences (ICEDSA, EBBT) as the standard tabular benchmark.
            </p>
          </div>

          <div style={{ background: '#f8fafc', padding: '14px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <CheckCircle2 size={18} color="#d97706" />
              <strong style={{ fontSize: '0.9rem' }}>Dataset Verification</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: '#64748b' }}>
              Confirmed 569 instances, 30 continuous nuclear boundary measurements extracted from digital scans of Fine Needle Aspirates (FNA), with 357 Benign and 212 Malignant ground-truth histology cases.
            </p>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', gap: '16px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            className={`btn ${filter === 'ALL' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setFilter('ALL')}
          >
            All Papers ({SCI_IEEE_EVIDENCE.length})
          </button>
          <button 
            className={`btn ${filter === 'SCI' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setFilter('SCI')}
          >
            SCI Indexed Journals
          </button>
          <button 
            className={`btn ${filter === 'IEEE' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setFilter('IEEE')}
          >
            IEEE Xplore Publications
          </button>
        </div>

        <div style={{ position: 'relative', minWidth: '300px' }}>
          <input
            type="text"
            className="input-field"
            placeholder="Search title, author, DOI, or proof..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ paddingLeft: '36px' }}
          />
          <Search size={18} color="#94a3b8" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }} />
        </div>
      </div>

      {/* Papers List */}
      <div>
        {filteredPapers.map((paper) => (
          <div key={paper.id} className="paper-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', flexWrap: 'wrap' }}>
              <div style={{ flex: 1 }}>
                <span className={`best-pill`} style={{ 
                  backgroundColor: paper.indexing.includes('SCI') ? '#eff6ff' : '#ecfdf5',
                  color: paper.indexing.includes('SCI') ? '#1e40af' : '#065f46',
                  borderColor: paper.indexing.includes('SCI') ? '#bfdbfe' : '#a7f3d0',
                  marginBottom: '8px'
                }}>
                  {paper.badgeType}
                </span>
                <h3 className="paper-title">{paper.title}</h3>
                <p className="paper-authors">{paper.authors} ({paper.year})</p>
                <p className="paper-venue">{paper.venue} &bull; {paper.volume}</p>
              </div>

              <div style={{ textAlign: 'right' }}>
                <span style={{ 
                  display: 'inline-block',
                  background: '#f1f5f9', 
                  padding: '4px 10px', 
                  borderRadius: '6px', 
                  fontSize: '0.8rem', 
                  fontWeight: 600,
                  color: '#334155',
                  marginBottom: '8px'
                }}>
                  {paper.indexing}
                </span>
                {paper.impactFactor !== 'IEEE Conference' && (
                  <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
                    Impact Factor: <strong>{paper.impactFactor}</strong>
                  </div>
                )}
              </div>
            </div>

            <div className="paper-proof-box">
              <strong style={{ display: 'block', color: '#0f172a', marginBottom: '4px' }}>
                🔬 Verifiable Dataset Proof in Paper:
              </strong>
              {paper.datasetProof}
            </div>

            <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
              <span style={{ fontSize: '0.82rem', color: '#64748b' }}>
                <strong>Key Significance:</strong> {paper.keyContribution}
              </span>
              <a 
                href={paper.url} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="btn btn-secondary"
                style={{ fontSize: '0.8rem', padding: '6px 12px' }}
              >
                View Paper DOI: {paper.doi}
                <ExternalLink size={14} />
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
