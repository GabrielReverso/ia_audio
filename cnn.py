import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES E ATIVAÇÃO DO CUDA ---

# **NOVO: Define o dispositivo globalmente**
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Ajuste estes caminhos conforme onde você salvou o dataset
CSV_PATH = 'ESC-50-master/meta/esc50.csv'
AUDIO_DIR = 'ESC-50-master/audio/'

# Parâmetros da FFT e da Rede
INPUT_SIZE = 1024  # Tamanho do vetor de entrada (bins de frequência)
BATCH_SIZE = 160
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_CLASSES = 50

# --- 1. CLASSE DO DATASET (Processamento FFT) ---
class ESC50Dataset(Dataset):
    def __init__(self, df, audio_dir, input_size=1024):
        self.df = df
        self.audio_dir = audio_dir
        self.input_size = input_size
        
        # Mapeamento de categoria (opcional, pois o csv já tem a coluna 'target')
        self.categories = df['category'].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['filename']
        label = row['target'] 
        
        file_path = os.path.join(self.audio_dir, file_name)

        # --- A Mágica da FFT acontece aqui ---
        try:
            # 1. Carregar áudio
            y, sr = librosa.load(file_path, sr=22050, duration=5.0)
            
            # Garantir que tem 5 segundos
            target_len = int(5.0 * sr)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]

            # 2. FFT e Magnitude
            fft_complex = np.fft.rfft(y)
            fft_mag = np.abs(fft_complex)

            # 3. Resize/Interpolação para caber na Input Layer da Rede
            x_old = np.linspace(0, 1, len(fft_mag))
            x_new = np.linspace(0, 1, self.input_size)
            features = np.interp(x_new, x_old, fft_mag)

            # 4. Normalização
            if features.max() > 0:
                features = features / features.max()
            
            # Converter para Tensor do PyTorch (float32)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # **MUDANÇA CRUCIAL PARA CNN1D:** Adicionar a dimensão do canal (1)
            # Formato esperado pela CNN1D: (channels, sequence_length) -> (1, INPUT_SIZE)
            features_tensor = features_tensor.unsqueeze(0) 
            
            return features_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Erro ao carregar {file_name}: {e}")
            return torch.zeros((1, self.input_size)), torch.tensor(label, dtype=torch.long)

# --- 2. MODELO DE REDE NEURAL (CNN1D) ---
class AudioCNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioCNN1D, self).__init__()
        
        # Camada 1D para extrair features locais da FFT
        self.conv_stack = nn.Sequential(
            # Input: (1 canal, 1024 features)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Reduz para 512
            nn.Dropout(0.2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Reduz para 256
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Reduz para 128
            nn.Dropout(0.2),
        )
        
        # O tamanho da saída antes da camada Linear precisa ser calculado:
        # 1024 -> 512 -> 256 -> 128 (após 3 MaxPools de tamanho 2)
        # O número de canais final é 64.
        FLAT_SIZE = 64 * (input_size // (2**3)) # 64 * 128 = 8192
        
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_SIZE, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x tem formato: (BATCH_SIZE, 1, 1024)
        x = self.conv_stack(x)
        # x tem formato: (BATCH_SIZE, 64, 128)
        logits = self.linear_stack(x)
        return logits

# --- 3. PREPARAÇÃO DOS DADOS ---
print("Carregando metadados...")
full_df = pd.read_csv(CSV_PATH)

# Separar Treino (80%) e Teste (20%)
train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['category'])

# Criar Datasets e DataLoaders
train_dataset = ESC50Dataset(train_df, AUDIO_DIR, INPUT_SIZE)
test_dataset = ESC50Dataset(test_df, AUDIO_DIR, INPUT_SIZE)

# Os DataLoaders permanecem os mesmos
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. CONFIGURAÇÃO DE TREINAMENTO ---

# **MUDANÇA: Instanciar a nova classe de modelo**
model = AudioCNN1D(INPUT_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"Modelo: AudioCNN1D\nParâmetros totais: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- 5. LOOP DE TREINAMENTO (Inalterado) ---
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Ajuste a frequência do print com base no novo BATCH_SIZE (400)
        if batch % 1 == 0: # Como o total de batches é pequeno, imprime todo batch
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval() 
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# --- 6. EXECUÇÃO ---
print("Iniciando treinamento...")
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, criterion, optimizer)
    test_loop(test_loader, model, criterion)

print("Treinamento concluído!")

# Salvar modelo
SAVE_PATH = "checkpoints/cnn/esc50_cnn1d_model.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Modelo salvo em: {SAVE_PATH}")