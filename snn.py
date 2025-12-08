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
from snntorch import functional as SF 
from snntorch import utils as SU 

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
NUM_WORKERS = 0 # Para I/O paralelo

# --- NOVOS PARÂMETROS SNN ---
NUM_STEPS = 25 # Duração da simulação SNN (Time Steps)
BETA = 0.95  # Constante de decaimento do neurônio Leaky (membrana)
GRAD = surrogate.atan() # Gradiente Substituto (arctan)

# --- 1. CLASSE DO DATASET (Processamento FFT) ---
# A classe do dataset permanece exatamente a mesma do código anterior (FFT feature)
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
            # Para SNN, a entrada precisa de uma dimensão extra para o tempo (Time Steps).
            # No entanto, vamos adicionar o time step diretamente no DataLoader/Loop
            return features_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Erro ao carregar {file_name}: {e}")
            return torch.zeros(self.input_size), torch.tensor(label, dtype=torch.long)

# --- 2. MODELO DE REDE NEURAL (SNN Densa) ---
class AudioSNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioSNN, self).__init__()
        
        # SNNs são construídas usando camadas lineares/convolucionais seguidas por neurônios spiking
        self.flat = nn.Flatten()
        
        # Camada 1: Linear + Leaky Neuron (LIF)
        self.fc1 = nn.Linear(input_size, 512)
        # O neurônio LIF (Leaky Integrate and Fire) com gradiente substituto
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=GRAD) 
        
        # Camada 2: Linear + Leaky Neuron
        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=GRAD)
        
        # Camada 3: Saída (não-spiking, para acúmulo de spikes)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Inicializa as variáveis de estado da membrana (mem)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Prepara um tensor para acumular as saídas da camada final
        spk_rec = torch.zeros(x.size(0), NUM_STEPS, NUM_CLASSES, device=device) 
        
        # Loop de simulação no tempo (Time Step)
        for step in range(NUM_STEPS):
            
            # Formato de entrada: x é (BATCH_SIZE, INPUT_SIZE)
            # A entrada da SNN precisa ser repetida ou codificada no tempo.
            # Aqui, simplesmente repetimos a feature FFT em cada Time Step (Taxa de Disparo Constante)
            cur = self.flat(x)
            
            # Camada 1
            cur = self.fc1(cur)
            spk1, mem1 = self.lif1(cur, mem1) # spk1 é o spike (0 ou 1)
            
            # Camada 2
            cur = self.fc2(spk1) # O input para a próxima camada é o spike da camada anterior
            spk2, mem2 = self.lif2(cur, mem2)
            
            # Camada de Saída
            cur = self.fc3(spk2)
            
            # Armazena a saída da última camada para acúmulo de erro
            spk_rec[:, step, :] = cur 

        # A saída final é a soma das saídas da última camada ao longo do tempo (spiking count)
        return spk_rec.sum(dim=1) 

# --- 3. PREPARAÇÃO DOS DADOS ---

# A diferença principal é que o SNN precisa de um loop de tempo.
# No entanto, a preparação dos dados (df, datasets) permanece inalterada.

# ... (Carregamento de metadados e split train/test) ...
print("Carregando metadados...")
full_df = pd.read_csv(CSV_PATH)
train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['category'])

train_dataset = ESC50Dataset(train_df, AUDIO_DIR, INPUT_SIZE)
test_dataset = ESC50Dataset(test_df, AUDIO_DIR, INPUT_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- 4. CONFIGURAÇÃO DE TREINAMENTO ---

model = AudioSNN(INPUT_SIZE, NUM_CLASSES).to(device)

# Otimizador e Scheduler permanecem os mesmos
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Loss Function: Usamos o CrossEntropyLoss, mas adaptado para SNN
# É comum usar o Mean Square Error (MSE) ou Cross Entropy adaptado para a soma de spikes
criterion = nn.CrossEntropyLoss()

# Função auxiliar para calcular acurácia na SNN
def accuracy_snn(spk_out, targets):
    # O output é a soma de spikes ao longo do tempo. 
    # Usamos argmax na soma para obter a classe com o maior número de spikes.
    return 100 * (spk_out.argmax(1) == targets).sum().item() / targets.size(0)

print(f"Modelo: AudioSNN\nParâmetros totais: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# --- 5. LOOP DE TREINAMENTO ---

def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass (executa o loop de tempo dentro do forward)
        spk_rec = model(X) # Saída é a soma dos logits acumulados
        
        # Loss: compara a soma dos outputs com o target
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


# -----------------------------------------------------------
# --- CORREÇÃO DE ERRO NO WINDOWS (Ponto de Entrada) ---
# -----------------------------------------------------------
if __name__ == '__main__':
    
    # --- 3. PREPARAÇÃO DOS DADOS (Necessário para num_workers > 0) ---
    # (Carregamento de metadados, split e criação de DataLoaders devem estar aqui)
    # ... (código de carregamento de metadados e DataLoaders acima)
    
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
    save_path = os.path.join(SAVE_DIR, "esc50_snn_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modelo SNN salvo em: {save_path}")