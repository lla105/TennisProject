import torch
model = torch.load('saved states/storke_classifier_weights.pth', map_location=torch.device('cpu'))
# model.eval()
print(model)