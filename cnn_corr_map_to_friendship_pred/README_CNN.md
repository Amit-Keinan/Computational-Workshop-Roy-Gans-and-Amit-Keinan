If your 90-length vector of between-friends ROI correlations has a meaningful ordering of ROIs (e.g., the atlas index is arranged spatially, or you can reorder them along a Hilbert curve like in the paper), then you can treat it like a 1D signal and use a 1D CNN.

The 1D CNN network learns local patterns in neighboring ROIs instead of treating all features as independent.

1D CNN idea for our case:
- Input shape: (batch_size, 90, 1) (90 in our case is the number of ROIs)
- 1 channel (just one value per ROI, based on correlation map value in range -1 and 1 (Pearson r-vals))
- Conv1D filters: slide across ROIs, looking for patterns in nearby ROIs.
- Pooling: reduces length while keeping features.
- Dense layers: combine extracted patterns to predict friendship distance.