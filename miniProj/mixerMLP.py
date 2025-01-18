import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    בלוק MLP בסיסי
    """
    def __init__(self, dim, hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)  # שכבה ראשונה
        self.gelu = nn.GELU()  # הפעלת GELU
        self.fc2 = nn.Linear(hidden_dim, dim)  # שכבה שנייה
        self.dropout = nn.Dropout(0.1)  # דרופאאוט למניעת אוברפיטינג

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MLPMixer(nn.Module):
    """
    מימוש של MLP-Mixer
    """
    def __init__(self, num_patches, num_channels, token_dim, channel_dim, num_classes, num_blocks):
        super(MLPMixer, self).__init__()
        # שכבת Embedding ראשונית
        self.embedding = nn.Linear(num_channels, channel_dim)
        
        # רשימת בלוקים של Mixer
        self.mixer_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channel_dim),  # נורמליזציה על הערוצים
                MLPBlock(num_patches, token_dim),  # Token-Mixing MLP
                nn.LayerNorm(channel_dim),  # נורמליזציה על הערוצים
                MLPBlock(channel_dim, channel_dim * 4)  # Channel-Mixing MLP
            )
            for _ in range(num_blocks)
        ])
        
        # ראש הסיווג
        self.head = nn.Sequential(
            nn.LayerNorm(channel_dim),  # נורמליזציה אחרונה
            nn.Linear(channel_dim, num_classes)  # סיווג סופי
        )
    
    def forward(self, x):
        # Embedding ראשוני
        x = self.embedding(x)  # גודל: [batch_size, num_patches, channel_dim]
        
        # מעבר דרך ה-Mixer Blocks
        for block in self.mixer_blocks:
            x = x + block(x)  # Residual Connection
        
        # ממוצע על פני כל הטלאים
        x = x.mean(dim=1)  # גודל: [batch_size, channel_dim]
        
        # סיווג סופי
        x = self.head(x)  # גודל: [batch_size, num_classes]
        return x


# דוגמה לשימוש:
batch_size = 8
num_patches = 196  # לדוגמה, 14x14 טלאים
num_channels = 512  # הקרנה ראשונית לממד 512
token_dim = 256  # ממד עבור Token Mixing
channel_dim = 512  # ממד עבור Channel Mixing
num_classes = 10  # מספר הקטגוריות לסיווג
num_blocks = 8  # מספר בלוקים של MLP-Mixer

# יצירת המודל
model = MLPMixer(num_patches, num_channels, token_dim, channel_dim, num_classes, num_blocks)

# דוגמה לכניסה (נתוני תמונה מוכנים)
dummy_input = torch.rand(batch_size, num_patches, num_channels)  # [batch_size, num_patches, num_channels]

# מעבר קדימה במודל
output = model(dummy_input)
print("Output shape:", output.shape)  # תוצאה: [batch_size, num_classes]
