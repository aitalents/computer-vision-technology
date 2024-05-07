import torchvision.transforms as transforms
import torch

def compute_scores(img_one, img_two, model):
    transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((224, 224))])
    img_one = img_one.convert('RGB')
    tensor_one = transform(img_one)
    emb_one = model(tensor_one.unsqueeze(0)).last_hidden_state[:, 0]
    img_two = img_two.convert('RGB')
    tensor_two = transform(img_two)
    emb_two = model(tensor_two.unsqueeze(0)).last_hidden_state[:, 0]
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.tolist()[0]