import torch
from PIL import Image
import torchvision.transforms as T
import hubconf

class xyz_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 

    def forward(self, tensor):
        ff = self.model(tensor)
        return ff
    

### Change your model here .dinov2_vits14() / .dinov2_vitb14() / .dinov2_vitl14() /.dinov2_vitg14() 
model = hubconf.dinov2_vits14()
mm = xyz_model(model).to('cpu')

mm.eval()
input_data = torch.randn(1, 3, 224, 224).to('cpu')
output = mm(input_data)

torch.onnx.export(mm, input_data, 'model.onnx', input_names = ['input'])