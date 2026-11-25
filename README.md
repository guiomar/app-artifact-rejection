# Power Spectral Density (PSD)

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Estimates MEG/EEG epoch rejection thresholds with autoreject and optionally repairs bad channels in epochs with autoreject or RANSAC (MNE-Python).**

Reads an MNE epochs file, optionally fits autoreject.AutoReject or autoreject.Ransac to repair or reject bad epochs, and computes a global peak-to-peak rejection dictionary with autoreject.get_rejection_threshold. Note: on this branch the code that saves the cleaned epochs and the reject-log figure is commented out, and the rejection dictionary is saved in out_dir (reject_dict.npy), so the registered outputs do not match what the code writes; the README describes a PSD App instead.

| | |
|---|---|
| 📥 **Inputs** | `neuro/meeg/mne/epochs` |
| 📤 **Outputs** | `neuro/meeg/mne/epochs` · `raw` · `generic/image/png` |
| 🐳 **Container** | `brainlife/mne:0.24dev.1` |
| ⚙️ **Resources** | 8 CPUs |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.585](https://doi.org/10.25663/brainlife.app.585) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `raw01` | `neuro/meeg/mne/epochs` (epochs) | yes | MNE epochs file (-epo.fif). |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `out_dir` | `neuro/meeg/mne/epochs` | Epoched data (equal length) |
| `our_dir` | `raw` |  |
| `out_figs` | `generic/image/png` |  |

<details>
<summary><b>⚙️ Parameters</b> (3)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `decim` | number | `1` | The decimation factor: Increment for selecting every nth time slice. TIP: We can use the `decim` parameter to only take every nth time slice. This speeds up the computation time. Note however that for low sampling rates and high decimation parameters, you might not detect "peaky artifacts" (with a fast timecourse) in your data. A low amount of decimation however is almost always beneficial at no decrease of accuracy. |
| `clean_epochs` | boolean | `true` | Return clean epochs (.fif file) in addition to the rejection dictionary |
| `method` | enum | `autoreject` | Repair method used when clean_epochs is true: autoreject (local thresholds) or ransac (bad channel detection). Options: `autoreject`, `ransac`. |

</details>

📚 **References used by this App:**
- Jas M, Engemann D, Raimondo F, Bekhti Y, Gramfort A. Automated rejection and repair of bad trials in MEG/EEG. PRNI (2016).
- Gramfort A, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci 7(267):1-13 (2013). [doi:10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

<!-- brainlife-app:end -->

[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.469-blue.svg)](https://doi.org/10.25663/brainlife.app.469)

This Brainlife App computes Power Spectral Density (PSD) on MEG files using MNE-Python Welch method with `mne.time_frequency.psd_welch` function

## Authors
- Guiomar Niso (guiomar.niso@ctb.upm.es)


### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your publications and code reusing this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://grantome.com/grant/NIH/R01-EB030896-01)

## Citations
We kindly ask that you cite the following articles when publishing papers and code using this code. 

*- brainlife.io Publishing and Apps:*  
Avesani, P., McPherson, B., Hayashi, S. et al. **The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services**. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

*- MNE-Python package:*  
Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T, Parkkonen L, and Hämäläinen MS.  
**MEG and EEG data analysis with MNE-Python**  
Frontiers in Neuroscience, 7(267):1–13, 2013. https://doi.org/10.3389/fnins.2013.00267
      
      

#### MIT Copyright (c) 2021 brainlife.io The University of Texas at Austin and Indiana University
