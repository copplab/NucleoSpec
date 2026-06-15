# NucleoSpec

**Nucleic Acid-Silver Complex & Cluster Analyzer**

A web-based application for analyzing nucleic acid (DNA/XNA)-silver complexes and nanoclusters from mass spectrometry data.

## Project Structure

```
NucleoSpec/
├── dna_silver_webapp.py    # Main Flask application
├── core/
│   └── analyzer.py         # DNASilverAnalyzer class (analysis logic)
├── lib/
│   └── pythoms/            # PythoMS library (isotope calculations)
├── templates/              # HTML templates
├── sample_data/            # Example spectrum files
└── environment_hf.yml      # HuggingFace deployment environment
```

## Installation

```bash
conda env create -f environment.yml
conda activate dna_mass_spec
python dna_silver_webapp.py
```

Open http://localhost:8080 in browser.

## Analysis Modes

### DNA-Ag<sub>N</sub> Mode
For single-stranded DNA-silver nanoclusters.
- Enter DNA sequence using A, T, G, C bases
- Automatically calculates molecular composition
- Uses N₀/Qcl framework for cluster characterization

### Ag(I)-DNA/XNA Complex Mode
For double-stranded DNA or DNA/XNA hybrids.
- **DNA Complex**: Enter two DNA sequences (Strand 1 and Strand 2)
- **XNA Complex**: Check "Use XNA" and enter two molecular formulas
  - Formulas are automatically combined by adding atoms

### XNA-Ag<sub>N</sub> Mode
For custom xeno nucleic acids (TNA, PNA, LNA, etc.).
- Enter XNA name for identification
- Enter complete molecular formula (e.g., C<sub>100</sub>H<sub>120</sub>N<sub>40</sub>O<sub>60</sub>P<sub>10</sub>)
- Optionally use JSME structure drawer to get formula

## Workflow

1. **Select Mode** - Choose DNA-Ag<sub>N</sub>, Ag(I)-DNA/XNA Complex, or XNA-Ag<sub>N</sub> from the mode selector
2. **Upload Spectrum** - Click "Choose File" and select your mass spectrum file
3. **Enter Information** - Provide DNA sequence or XNA formula based on selected mode
4. **Apply Settings** - Click the "Apply" button to confirm your settings
5. **Analyze Peaks** - Click any peak in the spectrum to find matching compositions
6. **Compare Results** - Toggle checkboxes to overlay theoretical isotope patterns on the experimental data

## File Format

Spectrum files should be two-column format (tab or comma separated):
```
m/z         intensity
1000.123    45678.9
1000.456    56789.0
1001.234    67890.1
```

Supported formats: .txt, .csv

## Output Fields

| Field | Description |
|-------|-------------|
| Formula | Neutral molecular formula of the cluster |
| Ion Formula | Charged species formula (can be copied) |
| n<sub>Ag <sub>| Number of silver atoms in the cluster |
| N₀ | Number of effective valence electrons|
| Qcl | Charge of inorganic core |
| z | Charge state of the detected ion |
| ΔX₀ | Difference between experimental and theoretical centroid (lower = better match). Primary criterion for Best Fit selection. |
| Pattern similarity | Mean of cosine similarity and Pearson correlation between experimental and theoretical isotope envelopes (0–1). Confidence indicator: ▲ > 0.8 high, ○ 0.5–0.8 moderate, ▽ < 0.5 low |

## Features

- **Charge Detection** - Automatic charge state determination from isotope spacing
- **Isotope Pattern Matching** - Compare experimental peaks with theoretical patterns
- **Adduct Support** - Account for common adducts (NH₄⁺, Na⁺, Cl⁻) plus user-defined custom adducts
- **Structure Drawing** - JSME molecule editor for drawing bioconjugate structures
- **Data Export** - Download theoretical spectra as CSV files

## Technical Details

- **Backend:** Python 3.12, Flask, NumPy, SciPy
- **Frontend:** HTML5, JavaScript, Plotly.js, JSME
- **Libraries:** PythoMS, IsoSpecPy

## Citation

If you use NucleoSpec in a publication, please cite:

> Lin, I.-H.; Copp, S. M. A Tutorial on Automated Mass Spectral Analysis using NucleoSpec for Compositional Assignment of Nucleic Acid–Silver Complexes and Nanoclusters. *ChemRxiv* 2026. [DOI: 10.26434/chemrxiv.15004738/v1](https://doi.org/10.26434/chemrxiv.15004738/v1)

## Support

- **Developer:** I-Hsin (Vivian) Lin
- **Email:** ihl1@uci.edu
- **Lab:** Copp Lab, University of California, Irvine
