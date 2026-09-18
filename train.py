import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
import time

# --- 1. DATA PREPROCESSING HELPERS ---
def calculate_distances(skeleton_data):
    """
    Input: (Frames, 17, 2)
    Output: (Frames, 78) - Euclidean distances between all pairs of 13 joints (head removed)
    """
    body_joints = skeleton_data[:, 5:17, :] # (Frames, 12, 2)
    joint17 = (skeleton_data[:, 5, :] + skeleton_data[:, 6, :]) / 2
    joints = np.concatenate([body_joints, joint17[:, np.newaxis, :]], axis=1) # (Frames, 13, 2)

    T, V, C = joints.shape
    distances = []
    for i in range(V):
        for j in range(i + 1, V):
            dist = np.linalg.norm(joints[:, i, :] - joints[:, j, :], axis=1)
            distances.append(dist)

    return np.array(distances).T # (72, 78)

class PDWalkDataset(Dataset):
    def __init__(self, csv_file, root_dir, cache_dir='processed_distances'):
        self.root_dir = root_dir
        self.data_list = []
        self.cache_dir = os.path.join(root_dir, "processed_distances")
        print("About to makedirs...", flush=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        print("Using cache_dir:", self.cache_dir, flush=True)

        df = pd.read_csv(csv_file, header=None)

        print(f"Checking/Preparing cache for {csv_file}...")
        for _, row in df.iterrows():
            folder_rel_path = row[0]
            label = int(row[1])
            folder_full_path = os.path.join(root_dir, folder_rel_path)

            if os.path.exists(folder_full_path):
                clips = [f for f in os.listdir(folder_full_path) if f.endswith('.npy')]
                for clip in clips:
                    raw_path = os.path.join(folder_full_path, clip)
                    cache_name = raw_path.replace(root_dir, "").replace("/", "_").replace("\\", "_")
                    cache_path = os.path.join(self.cache_dir, cache_name)

                    if not os.path.exists(cache_path):
                        raw_data = np.load(raw_path)
                        dist_data = calculate_distances(raw_data)
                        np.save(cache_path, dist_data)

                    self.data_list.append((cache_path, label))
        print("Dataset ready.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        cache_path, label = self.data_list[idx]
        dist_data = np.load(cache_path) # (72, 78)

        # ON-THE-FLY VELOCITY CALCULATION
        # np.diff reduces length by 1, so we prepend the first frame to maintain (72, 78)
        vel_data = np.diff(dist_data, axis=0, prepend=dist_data[:1, :])

        # Separate Standardization for better representational learning
        dist_data = (dist_data - np.mean(dist_data)) / (np.std(dist_data) + 1e-6)
        vel_data = (vel_data - np.mean(vel_data)) / (np.std(vel_data) + 1e-6)

        # Combine features: (72, 156)
        combined_features = np.concatenate([dist_data, vel_data], axis=1)

        return torch.FloatTensor(combined_features), torch.LongTensor([label]).squeeze()

# --- 2. MODEL ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerTower(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, use_pos_enc=True):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=1024, 
                                                   dropout=0.01,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ls_layer = nn.Linear(d_model, 200) 

    def forward(self, x):
        x = self.embedding(x)
        if self.use_pos_enc:
            x = self.pos_encoder(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1) 
        return F.relu(self.ls_layer(x)) # Internal ReLU for feature preservation

class TensorFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fso, fco):
        batch_size = fso.shape[0]
        ones = torch.ones(batch_size, 1).to(fso.device)
        fso_ext = torch.cat([fso, ones], dim=1) 
        fco_ext = torch.cat([fco, ones], dim=1) 
        fusion = torch.bmm(fso_ext.unsqueeze(2), fco_ext.unsqueeze(1))
        return fusion.view(batch_size, -1) 

class TwinTowerTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Step-wise input is now 156 (78 distances + 78 velocities)
        self.step_tower = TransformerTower(156, 512, 8, 4, use_pos_enc=True)
        # Channel-wise input is still 72 (sequence length)
        self.chan_tower = TransformerTower(72, 512, 8, 4, use_pos_enc=False)
        self.fusion = TensorFusion()

        fusion_dim = (200 + 1) ** 2
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, 200), 
            nn.ReLU(),
            nn.Linear(200, 2), 
            nn.Softmax(dim=1) # Softmax for confidence scores
        )

    def forward(self, x):
        # x: (Batch, 72, 156)
        fso = self.step_tower(x)

        # Transpose for Channel-wise: (Batch, 156, 72)
        x_chan = x.transpose(1, 2)
        fco = self.chan_tower(x_chan)

        f_fused = self.fusion(fso, fco)
        return self.decoder(f_fused)

# --- 3. TRAINING AND CROSS-VALIDATION LOOP ---

def run_cross_validation(project_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_base = os.path.join(project_root, 'datasets/CSVFolders/hrnet_pd_norm_natural_forward_2_class_5_fold/')
    weights_dir = os.getenv("WEIGHTS_DIR", "/workspace/weights")
    os.makedirs(weights_dir, exist_ok=True)

    results_log = []

    for fold in range(5):
        print(f"\n{'='*20} FOLD {fold} {'='*20}")

        train_csv = os.path.join(csv_base, f'train_{fold}.csv')
        test_csv = os.path.join(csv_base, f'test_{fold}.csv')

        train_ds = PDWalkDataset(train_csv, project_root)
        test_ds = PDWalkDataset(test_csv, project_root)

        train_loader = DataLoader(train_ds, batch_size=500, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=500, shuffle=False)

        model = TwinTowerTransformer().to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=1e-5)
        criterion = nn.NLLLoss()

        print(f"Training with {device}...")
        model.train()
        for epoch in range(400):
            total_loss = 0
            epoch_start=time.time()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                # Use log for NLLLoss since model ends in Softmax
                loss = criterion(torch.log(output + 1e-9), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Time:{time.time()-epoch_start:.2f}")

        # Testing
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f"Fold {fold} Results -> Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Log results for CSV
        results_log.append({
            'fold': fold,
            'accuracy': acc,
            'recall': rec,
            'f1_score': f1
        })

        torch.save(model.state_dict(), f"{weights_dir}/twin_tower_fold_{fold}.pth")

    # Save metrics to CSV in the weights directory
    df_metrics = pd.DataFrame(results_log)
    csv_path = os.path.join(weights_dir, "fold_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")
    print(f"Final Average Cross-Validation Accuracy: {df_metrics['accuracy'].mean():.4f}")

project_root = '/data'
run_cross_validation(project_root)