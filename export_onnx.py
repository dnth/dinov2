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
    

model = hubconf.dinov2_vits14()
mm = xyz_model(model).to('cpu')


# Set the model to evaluation mode
mm.eval()

# Generate some input data
input_data = torch.randn(1, 3, 224, 224).to('cpu')

# Pass the input data through the model
output = mm(input_data)

# Print the output
# print(output)


torch.onnx.export(mm, input_data, 'model.onnx', input_names = ['input'])