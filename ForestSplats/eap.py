import torch
import torch.nn as nn

class E_attr(nn.Module):
  def __init__(self, input_dim_a, output_nc=32):
    super(E_attr, self).__init__()
    dim = 64
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1),
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))  ## 1*1

  def forward(self, x):
    x = self.model(x)
    x = torch.squeeze(x)
    return x
  
# class directionModule(nn.Module):
#   def __init__(self, input_dim):
#     super(directionModule, self).__init__()
#     self.input_dim = input_dim
#     self.feature_dim1 = nn.Linear(input_dim, input_dim)
#     self.activation = nn.LeakyReLU()
#     self.feature_dim2 = nn.Linear(input_dim, input_dim)
  
#   def init_weights(self):
#     nn.init.uniform_(self.feature_dim1.weight, 0.0, 0.01)
#     nn.init.zeros_(self.feature_dim1.bias)
#     nn.init.uniform_(self.feature_dim2.weight, 0.0, 0.01)
#     nn.init.zeros_(self.feature_dim2.bias)
    
#   def forward(self, x):
#     self.feature_dim1(x)
#     self.activation(x)
#     self.feature_dim2(x)

#     return x
# # nn.Linear()

# class DirectionLoss(torch.nn.Module):
#   def __init__(self, loss_type='mse'):
#     super(DirectionLoss, self).__init__()
#     self.loss_type = loss_type
#     self.loss_func = { 'mse':    torch.nn.MSELoss, 'cosine': torch.nn.CosineSimilarity, 'mae':    torch.nn.L1Loss }[loss_type]()
#   def forward(self, x, y):
#     if self.loss_type == "cosine":
#       return 1. - self.loss_func(x, y)
#     return self.loss_func(x, y)

# class AppearanceModule(nn.Module):
#   def __init__(self, input_dim_a, input_dim, device):
#     super(AppearanceModule, self).__init__()
#     self.device = device
#     self.e_attr, self.d_module = E_attr(input_dim_a).to(self.device), directionModule(input_dim).to(self.device)
#     self.direction_loss = DirectionLoss("cosine")
  
#   def compute_direction_feature(self, image, embedding_intrinsic, transient_embedding):
#     fuse_feature = self.e_attr(image)
#     edit_direction = (fuse_feature - transient_embedding - embedding_intrinsic)
#     if edit_direction.sum() == 0:
#       fuse_feature = self.e_attr(fuse_feature + 1e-6)
#       edit_direction = (fuse_feature - embedding_intrinsic - transient_embedding)
#     embedding = self.d_module(edit_direction)
#     return embedding
  
#   def compute_appearance_feature(self, image):
#     appearance_feature = self.e_attr(image)
#     appearance_feature = self.d_module(appearance_feature)
#     return  appearance_feature
  
#   def clip_directional_loss(self, embedding: torch.Tensor, target_gt: torch.Tensor, embedding_intrinsic: torch.Tensor) -> torch.Tensor:
#     # print("embedding, self.e_attr(target_gt):    ", embedding.size(), self.e_attr(target_gt).size())
#     fuse_feature = self.e_attr(target_gt)
#     edit_direction = (fuse_feature - embedding_intrinsic)
#     target_feature = self.d_module(edit_direction)
#     return self.direction_loss(embedding.unsqueeze(1), target_feature.unsqueeze(1)).mean()    