- `rad_firstorder_Mean`  
  The average intensity value within the tumor region. It describes whether the region is overall brighter or darker.

- `rad_firstorder_Entropy`  
  A measure of how diverse or disordered the intensity values are in the tumor region. A higher value usually means greater intensity heterogeneity.

- `rad_firstorder_90Percentile`  
  The intensity value below which 90% of the pixels/voxels fall. It reflects the high-intensity level of the region while being less sensitive to extreme outliers than the maximum value.

- `rad_glcm_Contrast`  
  A texture feature from the Gray Level Co-occurrence Matrix (GLCM). It measures how large the intensity differences are between neighboring pixels. A higher value indicates stronger local variation.

- `rad_glcm_JointEntropy`  
  Another GLCM-based texture feature. It measures the complexity or randomness of spatial intensity relationships between neighboring pixels. A higher value suggests more complex and heterogeneous texture patterns.