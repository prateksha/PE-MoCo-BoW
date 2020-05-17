import torchvision.models as models
import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
resnet18 = models.resnet18(pretrained=True).to(device)
# print( resnet18.fc.in_features)
resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

# resnet18.fc = nn.Sequential(*list(resnet18.fc.children())[:-1])
# resnet18.avgpool = nn.Sequential(*list(resnet18.avgpool.children())[:-1])
# resnet18.bn2 = nn.Sequential(*list(resnet18.bn2.children())[:-1])
# print(resnet18)
root = "../Dataset/imagenet/imagenet_images/"
transform = T.Compose([T.Resize(256), 
                    T.CenterCrop(224), 
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

dataset = torchvision.datasets.ImageFolder(
    root, 
    transform=transform, 
    target_transform=None,  
    is_valid_file=None
    )

datasetloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=32,
                                            shuffle=False, 
                                            num_workers=2)

resnet18.eval()

def get_feature_maps():
    result_feature_map = []
    for data, label in datasetloader:
        data = data.to(device)
        # print(data.shape)
        feature = resnet18(data)
        # print(type(feature))
        # print(feature.shape)
        result_feature_map.append(feature)
    return result_feature_map

result_feature_map = get_feature_maps()

def get_stacked_feature_maps(result_feature_map):
    stacked_tensor = result_feature_map[0]
    for i in range(1, len(result_feature_map)):
      stacked_tensor = torch.cat([stacked_tensor, result_feature_map[i]], 0)
    return stacked_tensor

km = KMeans(
    n_clusters=2048, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

def get_kmeans_clusters(sampled_dataset):
    subset_size, channels, h, w = list(sampled_dataset.shape)
    x = torch.reshape(sampled_dataset, [subset_size*h*w, channels])
    # print(x.shape)
    y = km.fit_predict(x.cpu().detach().numpy())
    return y

