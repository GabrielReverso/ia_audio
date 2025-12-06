import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR # Importar o Scheduler

# --- CONFIGURAÇÕES E ATIVAÇÃO DO CUDA ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Ajuste estes caminhos conforme onde você salvou o dataset
CSV_PATH = 'ESC-50-master/meta/esc50.csv'
AUDIO_DIR = 'ESC-50-master/audio/'

# Parâmetros da FFT e da Rede
INPUT_SIZE = 1024 # Tamanho do vetor de entrada (bins de frequência)
BATCH_SIZE = 160
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_CLASSES = 50
NUM_WORKERS = 0

# --- 1. CLASSE DO DATASET (Processamento FFT) ---
class ESC50Dataset(Dataset):
    def __init__(self, df, audio_dir, input_size=1024):
        self.df = df
        self.audio_dir = audio_dir
        self.input_size = input_size
        self.categories = df['category'].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['filename']
        label = row['target'] 
        file_path = os.path.join(self.audio_dir, file_name)

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

            # 3. Resize/Interpolação
            x_old = np.linspace(0, 1, len(fft_mag))
            x_new = np.linspace(0, 1, self.input_size)
            features = np.interp(x_new, x_old, fft_mag)

            # 4. Normalização
            if features.max() > 0:
                features = features / features.max()
            
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            return features_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Erro ao carregar {file_name}: {e}")
            return torch.zeros(self.input_size), torch.tensor(label, dtype=torch.long)

# --- 2. MODELO DE REDE NEURAL (MLP OTIMIZADA) ---
class AudioLNN_Optimized(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioLNN_Optimized, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Camada 1: Batch Normalization (Estabilidade)
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            
            # Camada 2: Batch Normalization
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Camada de Saída
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# --- 5. LOOP DE TREINAMENTO ---
def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    model.train() 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Atualizar o scheduler no final da epoch
    scheduler.step()


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


if __name__ == '__main__':
    
    # --- 3. PREPARAÇÃO DOS DADOS ---
    print("Carregando metadados...")
    full_df = pd.read_csv(CSV_PATH)
    
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['category'])

    # Criar Datasets e DataLoaders
    train_dataset = ESC50Dataset(train_df, AUDIO_DIR, INPUT_SIZE)
    test_dataset = ESC50Dataset(test_df, AUDIO_DIR, INPUT_SIZE)

    # Criação do DataLoader DEVE estar dentro de if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 4. CONFIGURAÇÃO DE TREINAMENTO ---
    model = AudioLNN_Optimized(INPUT_SIZE, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Configurar o Scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 

    print(f"Modelo: AudioLNN_Optimized\nParâmetros totais: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # --- 6. EXECUÇÃO ---
    print("Iniciando treinamento...")
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, criterion, optimizer, scheduler)
        test_loop(test_loader, model, criterion)

    print("Treinamento concluído!")

    # Salvar modelo
    SAVE_DIR = 'checkpoints/lnn'
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "esc50_lnn_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modelo salvo em: {save_path}")