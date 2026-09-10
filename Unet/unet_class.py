import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):

    # __init__ contains the layers that contain learnable parameters

    def __init__(self, in_channels=1, out_channels=1, num_layers=4):           # in_channels=1 and out_channels=1 because we are detecting on a grayscale image
        super(UNet3D, self).__init__()
        self.num_layers = num_layers

        # Contracting path layers (Encoder)
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)         # we use padding=1 so that a 3x3 kernel has enough space to cover all pixels
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)

        # Expanding path layers (Decoder)                                         # we don't have to define BatchNorm3d again for each size, but it's clearer in my view 
        self.upconv6 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(512)
        self.upconv7 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(256)
        self.upconv8 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(128)
        self.upconv9 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm3d(64)

        # Output layer
        self.output = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        if self.num_layers == 4:
            return self.forward_4layers(x)
        else:
            return self.forward_3layers(x)

    def forward_4layers(self, x):
        # Contracting path (Encoder)
        conv1 = F.relu(self.bn1(self.conv1(x)))
        #print(f'conv1: {conv1.shape}')
        conv2 = F.relu(self.bn2(self.conv2(F.max_pool3d(conv1, kernel_size=2, stride=2)))) # each time we runf max_pool3d with kernel_size=2 we reduce the image size by half 
        #print(f'conv2: {conv2.shape}')
        conv3 = F.relu(self.bn3(self.conv3(F.max_pool3d(conv2, kernel_size=2, stride=2))))
        #print(f'conv3: {conv3.shape}')
        conv4 = F.relu(self.bn4(self.conv4(F.max_pool3d(conv3, kernel_size=2, stride=2))))
        #print(f'conv4: {conv4.shape}')
        conv5 = F.relu(self.bn5(self.conv5(F.max_pool3d(conv4, kernel_size=2, stride=2))))
        #print(f'conv5: {conv5.shape}')

        # Expanding path (Decoder)   
        upconv6 = self.upconv6(conv5) # upconv reverses the pooling operation

        conv6 = F.relu(self.bn6(self.conv6(torch.cat([upconv6, conv4], dim=1))))   # concatenate coarse features (upconv) with detailed ones (conv), then learn to integrate the coarse and detailed features using yet another conv 
        #print(f'conv6: {conv6.shape}')
        upconv7 = self.upconv7(conv6)
        #print(f'upconv7: {upconv7.shape}')
        conv7 = F.relu(self.bn7(self.conv7(torch.cat([upconv7, conv3], dim=1))))
        #print(f'conv7: {conv7.shape}')
        upconv8 = self.upconv8(conv7)
        #print(f'upconv8: {upconv8.shape}') 
        conv8 = F.relu(self.bn8(self.conv8(torch.cat([upconv8, conv2], dim=1))))
        #print(f'conv8: {conv8.shape}')
        upconv9 = self.upconv9(conv8)
        #print(f'upconv9: {upconv9.shape}') 
        conv9 = F.relu(self.bn9(self.conv9(torch.cat([upconv9, conv1], dim=1))))
        #print(f'conv9: {conv9.shape}')

        # Output layer
        output = self.output(conv9)

        return output


###########################################
############ Shallow net ##################
###########################################

    def forward_3layers(self, x):
        # Contracting path (Encoder)
        conv1 = F.relu(self.bn1(self.conv1(x)))
        #print(f'conv1: {conv1.shape}')
        conv2 = F.relu(self.bn2(self.conv2(F.max_pool3d(conv1, kernel_size=2, stride=2)))) # each time we runf max_pool3d with kernel_size=2 we reduce the image size by half 
        #print(f'conv2: {conv2.shape}')
        conv3 = F.relu(self.bn3(self.conv3(F.max_pool3d(conv2, kernel_size=2, stride=2))))
        #print(f'conv3: {conv3.shape}')
        conv4 = F.relu(self.bn4(self.conv4(F.max_pool3d(conv3, kernel_size=2, stride=2))))
        #print(f'conv4: {conv4.shape}')

        # Expanding path (Decoder)   
        upconv7 = self.upconv7(conv4) # upconv reverses the pooling operation

        conv7 = F.relu(self.bn7(self.conv7(torch.cat([upconv7, conv3], dim=1))))   # concatenate coarse features (upconv) with detailed ones (conv), then learn to integrate the coarse and detailed features using yet another conv 
        #print(f'conv6: {conv6.shape}')
        upconv8 = self.upconv8(conv7)
        #print(f'upconv7: {upconv7.shape}')
        conv8 = F.relu(self.bn8(self.conv8(torch.cat([upconv8, conv2], dim=1))))
        #print(f'conv7: {conv7.shape}')
        upconv9 = self.upconv9(conv8)
        #print(f'upconv8: {upconv8.shape}') 
        conv9 = F.relu(self.bn9(self.conv9(torch.cat([upconv9, conv1], dim=1))))
        #print(f'conv8: {conv8.shape}')

        # Output layer
        output = self.output(conv9)

        return output

