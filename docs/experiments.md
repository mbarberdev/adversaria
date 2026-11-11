##baseline 1

runs/baseline-001
cifar-10
45,000 examples
5,000 validation examples
split seed: 42
epochs: 10
best validation accuracy: 76.38%
checkpoint: runs/baseline-001/best.pt

observations:
peak of accuracy was epoch 8 but returns diminished after epoch 6
after epoch 8 the model had a clear overfitting divergence


##attack baseline 1

observations:

as epsilon increased, there were very clear plummets in accuracy
with the initial increase in epsilon (0.0078), Accuracy(clean)-Accuracy(attack)= 37.5%
Second pass of epsilon increase had the largest diminish at only 17.7% Accuracy(attack)