import torch
import torch.nn as nn

from training.utils.transformer import PositionalEncoding


class TimelineEncoder(nn.Module):
    def __init__(self, feature_dim, d_model, num_head=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model 

        self.proj_in = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_head, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_head, batch_first=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        frame_emb = self.proj_in(x)
        timeline_encoded = self.encoder(self.pos_encoder(frame_emb))

        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(query=query, key=timeline_encoded, value=timeline_encoded)
        return self.norm(pooled.squeeze(1))


class ChampionEncoder(nn.Module):
    def __init__(self, num_champions, champion_emb_dim=128, d_model=512, dropout=0.1):
        super().__init__()
        self.num_champions = num_champions
        self.d_model = d_model

        self.embedding_layer = nn.Embedding(num_champions, champion_emb_dim)
        self.proj_in = nn.Sequential(
            nn.Linear(champion_emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, champion_emb_dim)

    def forward(self, x):
        x = self.proj_in(self.embedding_layer(x))
        x = self.mlp(x) + x
        return self.proj_out(self.final_norm(x))
    

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim, num_head=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))  # (batch_size, num_champions (4-ally, 5-enemy), embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=num_head, batch_first=True)

    def forward(self, x):
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(query=query, key=x, value=x)
        return pooled.squeeze(1)


class ContextEncoder(nn.Module):
    def __init__(self, d_model, num_head=4, num_layers=2, use_attn_pooling=True):
        super().__init__()
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_head,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.ally_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.enemy_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ally_pool = AttentionPooling(d_model, num_head) if use_attn_pooling else lambda x: torch.mean(x, dim=1)
        self.enemy_pool = AttentionPooling(d_model, num_head) if use_attn_pooling else lambda x: torch.mean(x, dim=1)

        self.proj_out = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, ally_champ_emb, enemy_champ_emb):
        ally_out = self.ally_encoder(ally_champ_emb)
        enemy_out = self.enemy_encoder(enemy_champ_emb)

        ally_context = self.ally_pool(ally_out)
        enemy_context = self.enemy_pool(enemy_out)
  
        context = torch.cat([ally_context, enemy_context], dim=-1)
        return self.proj_out(context)


class Predictor(nn.Module):
    def __init__(self, d_model, use_FiLM=True):
        super().__init__()
        self.d_model = d_model
        self.use_FiLM = use_FiLM

        if self.use_FiLM:
            self.gamma_proj = nn.Linear(d_model, d_model)
            self.beta_proj = nn.Linear(d_model, d_model)
        else:
            self.proj_in = nn.Linear(d_model * 2, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),

            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, context_emb, conditional_emb):
        if self.use_FiLM:
            gamma = self.gamma_proj(conditional_emb)
            beta = self.beta_proj(conditional_emb)
            x = gamma * context_emb + beta
        else:
            # print(context_emb.shape)
            # print(conditional_emb.shape)
            x = self.proj_in(torch.cat([context_emb, conditional_emb], dim=-1))

        x = self.mlp(x)
        return self.proj_out(x)




