# LavakaMapping-v1

This is part of a project where I was attempting to first classify erosional gullies in the Madagascar highlands, and then measure phase correlation over gully features using Differential Interferometric Synthetic Aperature Radar (DInSAR).
\nFile breakdown:
\n<strong>PreProcess.py:</strong> Takes individual input features, converts them to the same EPSG, clips them to the same extent, resamples them to the same size, and stacks them to a single meta-file.
\n<strong>Trainer.py:</strong> Performs the model trainig. Uses a Random Forest Classifier from SK-Learn. Performs test-train split, cross-validation, and model analysis. Saves best fit model.
\n<strong>Classifier.py:</strong> Uses input model to classify points over stack of regional input features. 
\n<strong>Analysis.py:</strong> Samples phase correlation values from correlation interferogram for each land-cover type. Gathers general statistics and has a few plots it can produce.
\n<strong>RasterHelper.py:</strong> General library that contains most of the worker functions for the broad workflows.

I've called this v1 as the classifier could be significantly improved. There may be a v2 at some point that focuses on lavaka identification rather than correlation analysis.
Right now, this contains both workflows.
