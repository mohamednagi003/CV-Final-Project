import torch
from torchvision import transforms
from PIL import Image

# Load your classification model
def load_classification_model():
    model = torch.load("CV_Project/NoteBooks/Image_Classification_Final.ipynb", map_location=torch.device('cpu'))
    model.eval()
    return model

def classify_image(model, image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    predicted_class = output.argmax(1).item()
    return predicted_class
