import torch.nn as nn
import torch
import torchvision.models as models

class VideoCNNLSTM(nn.Module):
    def __init__(
        self, 
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
    ):
        super(VideoCNNLSTM, self).__init__()
        
        # Load pretrained ResNet18
        self.pretrained_model = models.squeezenet1_0(pretrained=True)
        
        # Optionally freeze ResNet weights
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Remove the final fully connected layer from ResNet
        # ResNet18 outputs 512 features from its last conv layer
        self.resnet = nn.Sequential(*list(self.pretrained_model.children())[:-1])  # Remove fc layer
        self.resnet.requires_grad = False
        
        # ResNet18 with default stride outputs 512 features
        # After 7x7 input becomes 1x1 (assuming input 224x224)
        self.lstm_input_size = 4608 # ResNet18's feature dimension
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.load_state_dict(torch.load("./model.pt"))

    def forward(self, x):
        # x shape: (batch_size, frames, channels, height, width)
        batch_size, frames, channels, height, width = x.size()
        
        # Process each frame with ResNet
        cnn_output = []
        for t in range(frames):
            # Get current frame
            frame = x[:, t, :, :, :]  # (batch_size, channels, height, width)
            
            # Apply ResNet
            frame = self.resnet(frame)  # (batch_size, 512, 1, 1)
            
            # Flatten ResNet output for LSTM
            frame = frame.view(batch_size, -1)  # (batch_size, 512)
            cnn_output.append(frame)
        
        # Stack CNN outputs to create sequence
        cnn_output = torch.stack(cnn_output, dim=1)  # (batch_size, frames, 512)
        
        # Feed sequence to LSTM
        lstm_out, _ = self.lstm(cnn_output)
        
        # Use final time step output for classification
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout and final classification
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        output = output.relu()
        output = self.fc2(output)
        
        return output
