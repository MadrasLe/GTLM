# --- START OF FILE TRITON (MENKE1)_V4_MONITORING.py ---

"""
MoE Transformer APENAS LIGER RMSNORM (CORRIGIDO, VETORIZADO E COM MONITORAMENTO)
🔥 FOCO: Adicionado cálculo de parâmetros ativos vs. totais e monitoramento de saúde dos gradientes.
✅ MANTIDO: LigerRMSNorm, Flash Attention 2, e MoE Vetorizado.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import time
import math
import json
import os
from datasets import Dataset as HFDataset
from torch.cuda.amp import autocast

from transformers import AutoTokenizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from deepspeed.ops.adam import DeepSpeedCPUAdam

# --- LIGER RMSNORM APENAS ---
try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    LIGER_RMSNORM_AVAILABLE = True
    print("✅ Liger RMSNorm DISPONÍVEL")
except ImportError:
    LIGER_RMSNORM_AVAILABLE = False
    print("⚠️ Liger RMSNorm não disponível. Usando nn.LayerNorm.")

# --- FLASH ATTENTION 2 ---
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN2_AVAILABLE = True
    print("✅ Flash Attention 2 DISPONÍVEL")
except ImportError:
    FLASH_ATTN2_AVAILABLE = False
    print("⚠️ Flash Attention 2 não disponível")

# --- HIPERPARÂMETROS ---
MAX_SEQ_LENGTH = 512
HIDDEN_DIM = 512
NUM_HEADS = 32
NUM_EXPERTS = 4
TOP_K = 2
NUM_LAYERS = 6
DROPOUT_RATE = 0.05
LEARNING_RATE = 1e-3
LOAD_BALANCING_ALPHA = 0.01
NUM_EPOCHS = 2
CHECKPOINT_EVERY_STEPS = 500
DRIVE_PATH = "/content/drive/MyDrive/moe_checkpoints_liger_rmsnorm"

# --- MoE EXPERT MANUAL (SEM LIGER SWIGLU) ---
class ManualSwiGLUExpert(nn.Module):
    def __init__(self, embed_dim, hidden_multiplier=5/3):
        super().__init__()
        hidden_dim = int(embed_dim * hidden_multiplier)
        hidden_base = 256
        hidden_dim = hidden_base * ((hidden_dim + hidden_base - 1) // hidden_base)
        
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=False) 
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        activated = self.activation(gate) * up
        return self.w_down(activated)

# --- MoE LAYER (VETORIZADA E CORRIGIDA) ---
class OptimizedMoELayer(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            ManualSwiGLUExpert(embed_dim) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        num_tokens = x_flat.shape[0]

        # 1. Gate computation
        with autocast(dtype=torch.bfloat16):
            logits = self.gate(x_flat.to(torch.bfloat16)).float()
        
        # 2. Top-k selection
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        # 3. Load balancing loss
        probs = F.softmax(logits, dim=-1)
        load_balance_loss = self.num_experts * (probs.mean(0) ** 2).sum()
        
        # 4. Aplanar os índices e criar um mapeamento para os tokens originais
        flat_topk_indices = topk_indices.view(-1)
        token_ids = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)

        # 5. Permutação (Shuffle): Agrupar tokens por expert de destino
        sorted_expert_indices, sorted_indices = flat_topk_indices.sort(0)
        permuted_tokens = x_flat[token_ids[sorted_indices]]
        permuted_weights = topk_weights.view(-1, 1)[sorted_indices]
        
        # 6. Processamento em Lote por Expert
        tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=self.num_experts)
        split_tokens = torch.split(permuted_tokens, tokens_per_expert.tolist())
        
        expert_outputs = []
        for i, expert_batch in enumerate(split_tokens):
            if expert_batch.numel() > 0:
                expert_outputs.append(self.experts[i](expert_batch))

        # 7. Reversão da Permutação (Un-shuffle)
        concatenated_outputs = torch.cat(expert_outputs)
        weighted_output = concatenated_outputs * permuted_weights
        output_flat = torch.zeros_like(x_flat)
        output_flat.index_add_(0, token_ids[sorted_indices], weighted_output.to(output_flat.dtype))
        
        return output_flat.view(B, S, D), load_balance_loss, probs.mean(0)

# --- MANUAL ROPE ---
class ManualRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
        
    def forward(self, x):
        B, S, H, D = x.shape
        cos = self.cos[:S].unsqueeze(0).unsqueeze(2)
        sin = self.sin[:S].unsqueeze(0).unsqueeze(2)
        x_reshaped = x.reshape(B, S, H, D // 2, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        return torch.stack((y_even, y_odd), dim=-1).reshape(B, S, H, D)

# --- ATTENTION COM ROPE MANUAL ---
class OptimizedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = ManualRotaryEmbedding(self.head_dim, MAX_SEQ_LENGTH)
        
    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim) 
        v = v.view(B, S, self.num_heads, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        
        if FLASH_ATTN2_AVAILABLE:
            attn_out = flash_attn_func(q, k, v, dropout_p=0.0 if not self.training else DROPOUT_RATE, causal=True, return_attn_probs=False)
            attn_out = attn_out.contiguous().view(B, S, D)
        else: 
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0 if not self.training else DROPOUT_RATE)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.o_proj(attn_out)

# --- TRANSFORMER BLOCK COM LIGER RMSNORM ---
class OptimizedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts, top_k):
        super().__init__()
        self.attn = OptimizedAttention(embed_dim, num_heads)
        self.moe = OptimizedMoELayer(embed_dim, num_experts, top_k)
        
        if LIGER_RMSNORM_AVAILABLE:
            self.ln1 = LigerRMSNorm(embed_dim)
            self.ln2 = LigerRMSNorm(embed_dim)
        else:
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        residual = x
        x_norm = self.ln1(x)
        x = residual + self.attn(x_norm)
        
        residual = x
        x_norm = self.ln2(x)
        moe_out, lb_loss, expert_dist = self.moe(x_norm)
        x = residual + moe_out
        
        return x, lb_loss, expert_dist

# --- MODELO PRINCIPAL ---
class OptimizedMoEModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=HIDDEN_DIM, num_heads=NUM_HEADS, 
                 num_experts=NUM_EXPERTS, top_k=TOP_K, num_layers=NUM_LAYERS):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.layers = nn.ModuleList([
            OptimizedTransformerBlock(embed_dim, num_heads, num_experts, top_k) for _ in range(num_layers)
        ])
        
        if LIGER_RMSNORM_AVAILABLE: self.ln_f = LigerRMSNorm(embed_dim)
        else: self.ln_f = nn.LayerNorm(embed_dim)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None: torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        total_lb_loss = 0.0
        expert_metrics = []
        
        for layer in self.layers:
            x, lb_loss, expert_dist = layer(x)
            total_lb_loss += lb_loss
            expert_metrics.append(expert_dist)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        lm_loss = None
        if labels is not None:
            lm_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))

        avg_expert_dist = torch.stack(expert_metrics).mean(0) if expert_metrics else None
        avg_lb_loss = total_lb_loss / len(self.layers) if self.layers else 0.0
        
        return lm_loss, avg_lb_loss, avg_expert_dist

# --- DEEPSPEED CONFIG ---
OPTIMIZED_DEEPSPEED_CONFIG = {
    "train_batch_size": 1024, "train_micro_batch_size_per_gpu": 32, "gradient_accumulation_steps": 32, "steps_per_print": 25, "bf16": {"enabled": True},
    "zero_optimization": { "stage": 1, "offload_optimizer": {"device": "cpu", "pin_memory": True}, "overlap_comm": True, "contiguous_gradients": True, "reduce_scatter": True, "reduce_bucket_size": "auto", "allgather_bucket_size": "auto", "round_robin_gradients": True },
    "gradient_clipping": 1.0, "activation_checkpointing": { "partition_activations": True, "cpu_checkpointing": True, "contiguous_memory_optimization": True }, "communication_data_type": "bf16"
}

# --- DATASET ---
class CustomArrowDataset(Dataset):
    def __init__(self, dataset_folder):
        import glob
        arrow_files = glob.glob(os.path.join(dataset_folder, "*.arrow"))
        if not arrow_files: raise ValueError(f"Nenhum arquivo .arrow encontrado em {dataset_folder}")
        from datasets import concatenate_datasets
        datasets = [HFDataset.from_file(file) for file in arrow_files]
        self.dataset = concatenate_datasets(datasets)
        print(f"✅ Dataset carregado: {len(self.dataset)} samples de {len(arrow_files)} arquivos")
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return {'input_ids': torch.tensor(self.dataset[idx]['input_ids'])}

# --- CHECKPOINTS ---
def save_checkpoint(model_engine, epoch, step, loss, save_path):
    tag = f"global_step{step}"
    client_state = {'epoch': epoch, 'step': step, 'loss': loss}
    model_engine.save_checkpoint(save_path, tag=tag, client_state=client_state)
    print(f"💾 Checkpoint salvo: {os.path.join(save_path, tag)}")

def load_checkpoint(model_engine, checkpoint_path):
    load_path, client_state = model_engine.load_checkpoint(checkpoint_path)
    if load_path:
        epoch = client_state.get('epoch', 0)
        step = client_state.get('step', 0)
        loss = client_state.get('loss', float('inf'))
        print(f"📂 Checkpoint carregado: {load_path}\n   -> Retomando Época {epoch}, Passo {step}")
        return epoch, step, loss
    print("⚠️ Nenhum checkpoint encontrado.")
    return 0, 0, float('inf')

# --- GRADIENT MONITORING (LEVE) ---
def monitor_gradients_health(model_engine, global_step):
    grad_norm = model_engine.get_global_grad_norm() if hasattr(model_engine, 'get_global_grad_norm') else None
    if grad_norm is not None:
        if torch.isinf(grad_norm) or torch.isnan(grad_norm):
            print(f"\n🔴 ALERTA (Passo {global_step}): Gradientes explodindo (NaN/inf)! Norma: {grad_norm}")
        elif grad_norm < 1e-6:
            print(f"\n🟡 AVISO (Passo {global_step}): Gradientes desaparecendo! Norma: {grad_norm:.6f}")
        else:
            print(f"\n✅ (Passo {global_step}) Saúde dos Gradientes: OK. Norma Global: {grad_norm:.4f}")
    return grad_norm

# --- INFO ---
def print_model_info(model, tokenizer):
    # --- NOVA LÓGICA DE CONTAGEM DE PARÂMETROS ---
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calcular parâmetros de um único expert
    params_per_expert = sum(p.numel() for p in model.layers[0].moe.experts[0].parameters())
    
    # Calcular parâmetros totais de todos os experts em todas as camadas
    total_expert_params = params_per_expert * NUM_EXPERTS * NUM_LAYERS
    
    # Parâmetros compartilhados (não-experts)
    shared_params = total_params - total_expert_params
    
    # Parâmetros ativos = compartilhados + (top_k experts por camada)
    active_params = shared_params + (params_per_expert * TOP_K * NUM_LAYERS)
    
    utilization_ratio = (active_params / total_params) * 100

    print("\n" + "="*80)
    print(f"🚀 MoE COM LIGER RMSNORM E MoE VETORIZADO")
    print("="*80)
    print(f"Arquitetura: {NUM_LAYERS} layers | {NUM_EXPERTS} experts | Top-{TOP_K}")
    print(f"Parâmetros Totais:   {total_params/1e9:.3f}B ({total_params:,})")
    print(f"Parâmetros Ativos:   {active_params/1e9:.3f}B ({active_params:,})")
    print(f"Taxa de Utilização:  {utilization_ratio:.2f}% dos parâmetros totais por token")
    print("-" * 80)
    print(f"🔥 OTIMIZAÇÕES:")
    print(f"  ├─ MoE Layer: ✅ Vetorizada (Token Shuffling)")
    print(f"  ├─ Liger RMSNorm: {'✅ Enabled' if LIGER_RMSNORM_AVAILABLE else '❌ Disabled'}")
    print(f"  ├─ Manual SwiGLU: ✅ Stable")
    print(f"  ├─ Manual RoPE: ✅ Stable") 
    print(f"  └─ Standard CrossEntropy: ✅ Reliable")
    print(f"Flash Attention: {'✅' if FLASH_ATTN2_AVAILABLE else '❌'}")
    print("="*80)

# --- TRAINING PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    
    TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
    DATASET_PATH = "/content/dataset/train"  # MUDE PARA SEU DATASET
    
    os.makedirs(DRIVE_PATH, exist_ok=True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"): torch.backends.cuda.enable_flash_sdp(True)
    
    print(f"🔧 Carregando tokenizer '{TOKENIZER_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer carregado: {tokenizer.vocab_size:,} tokens")

    train_dataset = CustomArrowDataset(DATASET_PATH)
    model = OptimizedMoEModel(vocab_size=tokenizer.vocab_size)
    print_model_info(model, tokenizer)
    
    if args.compile:
        print("🔥 Aplicando torch.compile...")
        model = torch.compile(model, mode="max-autotune")

    train_loader = DataLoader(train_dataset, batch_size=OPTIMIZED_DEEPSPEED_CONFIG['train_micro_batch_size_per_gpu'], shuffle=True, num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=4, persistent_workers=True)
    
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    OPTIMIZED_DEEPSPEED_CONFIG["train_batch_size"] = (OPTIMIZED_DEEPSPEED_CONFIG["train_micro_batch_size_per_gpu"] * OPTIMIZED_DEEPSPEED_CONFIG["gradient_accumulation_steps"] * world_size)
    
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    
    steps_per_epoch = max(1, len(train_loader) // OPTIMIZED_DEEPSPEED_CONFIG['gradient_accumulation_steps'])
    total_global_steps = max(1, steps_per_epoch * NUM_EPOCHS)
    warmup_steps = max(1, int(0.03 * total_global_steps))
    cosine_steps = max(1, total_global_steps - warmup_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=LEARNING_RATE*0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    model_engine, optimizer, _, scheduler = deepspeed.initialize(args=args, model=model, optimizer=optimizer, lr_scheduler=scheduler, config=OPTIMIZED_DEEPSPEED_CONFIG)
    
    start_epoch, global_step, best_loss = 0, 0, float('inf')
    if args.resume_checkpoint: start_epoch, global_step, best_loss = load_checkpoint(model_engine, args.resume_checkpoint)
    
    model_engine.train()
    start_time = time.time()
    total_tokens = 0
    recent_losses = []
    
    if torch.cuda.is_available(): 
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print("\n🚀 INICIANDO TREINO COM MoE VETORIZADO E MONITORAMENTO...")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=(model_engine.local_rank != 0))
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(model_engine.device, non_blocking=True).long()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_input_ids = input_ids[:, :-1].contiguous()
            pad_mask = shift_labels != tokenizer.pad_token_id
            shift_labels_masked = shift_labels.masked_fill(~pad_mask, -100)
            
            lm_loss, lb_loss, expert_dist = model_engine(shift_input_ids, shift_labels_masked)
            
            if lm_loss is None or torch.isnan(lm_loss) or torch.isinf(lm_loss):
                if model_engine.local_rank == 0: print(f"\n⚠️ Passo {global_step} pulado devido a loss inválida: {lm_loss}\n")
                model_engine.zero_grad()
                continue
            
            loss = lm_loss + LOAD_BALANCING_ALPHA * lb_loss
            model_engine.backward(loss)
            
            # --- MONITORAMENTO DE GRADIENTES ANTES DO STEP ---
            grad_norm = model_engine.get_global_grad_norm() if hasattr(model_engine, 'get_global_grad_norm') else 0.0
            
            model_engine.step()
            
            total_tokens += pad_mask.sum().item()
            recent_losses.append(lm_loss.item())
            
            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                
                if model_engine.local_rank == 0 and global_step % 25 == 0:
                    avg_loss = sum(recent_losses[-10:]) / min(10, len(recent_losses))
                    elapsed = time.time() - start_time
                    tps = total_tokens / elapsed if elapsed > 0 else 0
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # --- NOVA BARRA DE PROGRESSO COM GRAD NORM ---
                    progress_bar.set_postfix({
                        "Loss": f"{avg_loss:.3f}",
                        "LR": f"{current_lr:.2e}", 
                        "GradNorm": f"{grad_norm:.3f}",
                        "TPS": f"{tps:,.0f}",
                        "GPU": f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                    })
                    
                    # --- VERIFICA A SAÚDE DOS GRADIENTES ---
                    if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                        print(f"\n🔴 ALERTA: Gradientes explodindo (NaN/inf)!")
                    elif grad_norm < 1e-6 and global_step > 100: # Evita falso positivo no início
                        print(f"\n🟡 AVISO: Gradientes desaparecendo!")
                
                if global_step > 0 and global_step % CHECKPOINT_EVERY_STEPS == 0 and model_engine.local_rank == 0:
                    current_loss = sum(recent_losses[-20:]) / min(20, len(recent_losses))
                    save_checkpoint(model_engine, epoch, global_step, current_loss, DRIVE_PATH)
        
        if model_engine.local_rank == 0:
            current_loss = sum(recent_losses[-len(progress_bar):]) / len(progress_bar) if recent_losses else 0
            print(f"\n📊 Epoch {epoch+1} Completa - Loss: {current_loss:.4f}")
            save_checkpoint(model_engine, epoch, global_step, current_loss, DRIVE_PATH)
    
    total_time = time.time() - start_time
    if model_engine.local_rank == 0:
        final_tps = total_tokens / total_time if total_time > 0 else 0
        print("\n" + "="*70)
        print("🏆 TREINO COMPLETO - MoE VETORIZADO!")
        print("="*70)
        print(f"⏱️ Tempo: {total_time/3600:.2f} horas")
        print(f"🚀 TPS Final: {final_tps:,.0f}")
        print(f"💾 GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        print(f"🔥 MoE Type: {'Vetorizado (Token Shuffling)'}")
        print("="*70)

if __name__ == "__main__":
    main()