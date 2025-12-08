import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR 

# Importar bibliotecas SNN
import snntorch as snn 
from snntorch import surrogate 

# --- CONFIGURAÇÕES E ATIVAÇÃO DO CUDA ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Ajuste estes caminhos conforme onde você salvou o dataset
CSV_PATH = 'ESC-50-master/meta/esc50.csv'
AUDIO_DIR = 'ESC-50-master/audio/'

# --- NOVAS CONFIGURAÇÕES DE CLASSE ---
# CLASSES_TO_LOAD: Lista com as classes desejadas
CLASSES_TO_LOAD = ['chirping_birds', 'dog', 'cat', "cow", "sheep"] 

# Parâmetros da FFT e da Rede
INPUT_SIZE = 1024 # Tamanho do vetor de entrada (bins de frequência)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_CLASSES = len(CLASSES_TO_LOAD) 
NUM_WORKERS = 0 # Para I/O paralelo

# --- NOVOS PARÂMETROS SNN ---
NUM_STEPS = 25 # Duração da simulação SNN (Time Steps)
BETA = 0.95  # Constante de decaimento do neurônio Leaky (membrana)
GRAD = surrogate.atan() # Gradiente Substituto (arctan)

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
        # O rótulo 'target' agora é 0, 1 ou 2 graças à re-indexação no main
        label = row['target'] 
        file_path = os.path.join(self.audio_dir, file_name)

        try:
            y, sr = librosa.load(file_path, sr=22050, duration=5.0)
            target_len = int(5.0 * sr)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]

            fft_complex = np.fft.rfft(y)
            fft_mag = np.abs(fft_complex)

            x_old = np.linspace(0, 1, len(fft_mag))
            x_new = np.linspace(0, 1, self.input_size)
            features = np.interp(x_new, x_old, fft_mag)

            if features.max() > 0:
                features = features / features.max()
            
            features_tensor = torch.tensor(features, dtype=torch.float32)
            return features_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Erro ao carregar {file_name}: {e}")
            return torch.zeros(self.input_size), torch.tensor(label, dtype=torch.long)

# --- 2. MODELO DE REDE NEURAL (SNN Densa) ---
class AudioSNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioSNN, self).__init__()
        
        self.flat = nn.Flatten()
        
        # Camada 1
        self.fc1 = nn.Linear(input_size, 512)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=GRAD) 
        
        # Camada 2
        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=GRAD)
        
        # Camada 3: Saída (num_classes = 3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk_rec = torch.zeros(x.size(0), NUM_STEPS, NUM_CLASSES, device=device) 
        
        # Loop de simulação no tempo (Time Step)
        for step in range(NUM_STEPS):
            cur = self.flat(x)
            
            # Camada 1
            cur = self.fc1(cur)
            spk1, mem1 = self.lif1(cur, mem1) 
            
            # Camada 2
            cur = self.fc2(spk1) 
            spk2, mem2 = self.lif2(cur, mem2)
            
            # Camada de Saída
            cur = self.fc3(spk2)
            
            # Armazena a saída da última camada 
            spk_rec[:, step, :] = cur 

        # A saída final é a soma das saídas da última camada ao longo do tempo
        return spk_rec.sum(dim=1) 

# --- 3. FUNÇÕES DE TREINAMENTO E TESTE ---

# Função auxiliar para calcular acurácia na SNN
def accuracy_snn(spk_out, targets):
    return 100 * (spk_out.argmax(1) == targets).sum().item() / targets.size(0)

def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        spk_rec = model(X) 
        loss = loss_fn(spk_rec, y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss_val, current = loss.item(), batch * len(X)
            acc = accuracy_snn(spk_rec, y)
            print(f"loss: {loss_val:>7f} | Acc: {acc:>4.1f}% [{current:>5d}/{size:>5d}]")

    scheduler.step()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            spk_rec = model(X)
            
            test_loss += loss_fn(spk_rec, y).item()
            correct += (spk_rec.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    
# --- 4. PREPARAÇÃO DOS DADOS (MODIFICADA PARA FILTRAGEM E RE-INDEXAÇÃO) ---
print("Carregando metadados e filtrando classes...")
full_df = pd.read_csv(CSV_PATH)

# 1. Filtra o DataFrame para incluir apenas as classes desejadas
filtered_df = full_df[full_df['category'].isin(CLASSES_TO_LOAD)].copy()

# 2. Mapeia e re-indexa os rótulos ('target') para 0, 1, 2...
# Isto é crucial para o treinamento com menos classes.
category_to_target = {category: i for i, category in enumerate(CLASSES_TO_LOAD)}
filtered_df['target'] = filtered_df['category'].map(category_to_target)

print(f"Dataset filtrado: {len(filtered_df)} amostras.")
print(f"Mapeamento de rótulos: {category_to_target}")

# 3. Split treino/teste
train_df, test_df = train_test_split(
    filtered_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=filtered_df['target'] # Estratifica pelo novo rótulo 'target'
)

train_dataset = ESC50Dataset(train_df, AUDIO_DIR, INPUT_SIZE)
test_dataset = ESC50Dataset(test_df, AUDIO_DIR, INPUT_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- 5. CONFIGURAÇÃO DE TREINAMENTO ---
model = AudioSNN(INPUT_SIZE, NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

print(f"Modelo: AudioSNN\nParâmetros totais: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# --- 6. EXECUÇÃO ---
print("Iniciando treinamento SNN...")
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, criterion, optimizer, scheduler)
    test_loop(test_loader, model, criterion)

print("Treinamento SNN concluído!")

# Salvar modelo
SAVE_DIR = 'checkpoints/snn'
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, "esc50_snn_model_class_selection.pth")
torch.save(model.state_dict(), save_path)
print(f"Modelo SNN salvo em: {save_path}")