from Models.PC.pc_nn import pc_net
from Models.PC.pc_layers import *
import torchvision as tv
import torchvision.transforms as transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn, loss_fn_deriv = parse_loss_function("crossentropy")
batch_size = 64
lr = 0.005
n_inference_steps = 50
inference_lr = 0.005


l1 = ConvLayer(input_size=28, num_channels=1, num_filters=6, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv)

# Max pooling layer with kernel size 2x2
l2 = MaxPool(2, device=DEVICE)

# Convolutional layer with input size 14x14 (after max pooling), 6 input channels, 16 output filters, kernel size 5x5
l3 = ConvLayer(input_size=12, num_channels=6, num_filters=16, batch_size=batch_size, kernel_size=5, learning_rate=lr, f=relu, df=relu_deriv)

# Projection layer with input size corresponding to the output size of the previous conv layer, 16x5x5
l4 = ProjectionLayer(input_size=(64, 16, 8, 8), output_size=120, f=relu, df=relu_deriv, learning_rate=lr)

# Fully connected layer
l5 = FCLayer(input_size=120, output_size=84, batch_size=64, learning_rate=lr, f = relu, df = relu_deriv)

# Final fully connected layer with 10 output classes for MNIST
l6 = FCLayer(input_size=84, output_size=10, batch_size=64, learning_rate=lr, f = F.softmax, df= linear_deriv)

# List of layers
layers = [l1, l2, l3, l4, l5, l6]

mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./mnist_data', train=True,
                                        download=True, transform=mnist_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                    shuffle=True)
train_data = list(iter(trainloader))
testset = tv.datasets.MNIST(root='./mnist_data', train=False,
                                        download=True, transform=mnist_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                    shuffle=True)
test_data = list(iter(testloader))


net = pc_net(layers, n_inference_steps,
             inference_lr,
             loss_fn = loss_fn, loss_fn_deriv = loss_fn_deriv,
             device=DEVICE)

net.train(train_data[0:-2],test_data[0:-2],
          5, n_inference_steps,
          '/Users/emmagraham/Documents/GitHub/propagation/CL/pc_saves',
          '/Users/emmagraham/Documents/GitHub/propagation/CL/pc_saves', 
          'None')