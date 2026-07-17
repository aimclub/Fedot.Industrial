# EEG Examples

Purpose: EEG classification scenarios. The current API exposes
`eeg_classification` with kernel-learning and optional PDL model specs. Local
`.npy` arrays and prediction CSVs stay outside git; use
`../../external_data_manifest.json` for external data delivery.

Tracked notebooks are reproducible current-API previews. Raw MNE signal
processing remains an extended local workflow because it requires optional EEG
dependencies and external signal files.
