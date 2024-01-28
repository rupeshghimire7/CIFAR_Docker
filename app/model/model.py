import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image




# VGG MODEL ARCHITECTURE
class VGG(nn.Module):
    def __init__(
        self,
        architecture,
        in_channels=3,
        in_height=224,
        in_width=224,
        num_hidden=4096,
        num_classes=1000
    ):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.convs = self.init_convs(architecture)
        self.fcs = self.init_fcs(architecture)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return x

    def init_fcs(self, architecture):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = self.in_height // factor
        out_width = self.in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        return nn.Sequential(
            nn.Linear(
                last_out_channels * out_height * out_width,
                self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_classes),
            nn.ReLU()
        )

    def init_convs(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )

        return nn.Sequential(*layers)









# VGG ARCHITECTURES TYPES
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}







def get_model():
    # VGG19 model instantiated
    model = VGG(
        in_channels=3,
        in_height=32,
        in_width=32,
        architecture=VGG_types["VGG19"]
    )

    # Load the saved model state dict
    checkpoint_path = "./model/checkpoint/ckpt.pth"
    checkpoint = torch.load(checkpoint_path)



    # Check if the model was saved using DataParallel
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        # Remove the 'module.' prefix from the keys
        state_dict = {key.replace('module.', ''): value for key, value in checkpoint['model_state_dict'].items()}
    else:
        state_dict = checkpoint['model_state_dict']

    # Load the model state_dict
    model.load_state_dict(state_dict)

    # Load other checkpoint information
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    model.eval()
    return model


# Initialize model
model = get_model()


# INFERENCE FUNCTION
def inference(path_to_image):
    # Define image transformations
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Load and preprocess the image
    image = Image.open(path_to_image).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Optionally, apply softmax if your model outputs raw logits
    probabilities = nn.functional.softmax(output[0], dim=0)

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities[predicted_class].item()




def make_inference(path_to_img):
    path_to_image = path_to_img
    return inference(path_to_image)




# specify version of your model
__version__ = '0.1.0'