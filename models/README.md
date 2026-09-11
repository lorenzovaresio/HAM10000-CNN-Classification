# Trained Models

Five models were trained using 5-fold group cross-validation.

For each fold, the model achieving the highest validation balanced accuracy was saved as:

- `best_model_fold_1.pth`
- `best_model_fold_2.pth`
- `best_model_fold_3.pth`
- `best_model_fold_4.pth`
- `best_model_fold_5.pth`

The model weights are not stored directly in the Git repository because of their file size.

All models can be reproduced using the training pipeline provided in the project notebook.