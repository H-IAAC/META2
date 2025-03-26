import torch
from torch import nn
import torch.nn.functional as F
import math
import loralib as lora
import gc

class BaseConvolutionalModel(nn.Module):

    def __init__(self, height, width, output_classes):
        super(BaseConvolutionalModel, self).__init__()
        self.conv = nn.Sequential(
            self.make_conv_block(1, 8),
            # -2
            self.make_conv_block(8, 16),
            # -4
            self.make_conv_block(16, 32),
            # -6
            self.make_conv_block(32, 64),
            # -8
            nn.Flatten(),
            nn.Linear(in_features=64*(height-8)*(width-8), out_features=256),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=256, out_features=output_classes)
        )

    def make_conv_block(self, input_channels, output_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(torch.unsqueeze(x, 1))


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_shape, hidden_layer_dimensions, num_classes):
        super(FullyConnectedNetwork, self).__init__()

        flatten_size = 1
        for dim in input_shape:
            flatten_size *= dim

        layers = [nn.Flatten()]
        in_features = flatten_size
        for out_features in hidden_layer_dimensions:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features

        layers.append(nn.Linear(in_features, num_classes))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.wq = torch.nn.Parameter(torch.randn((n_embd, head_size)))
        self.wk = torch.nn.Parameter(torch.randn((n_embd, head_size)))
        self.wv = torch.nn.Parameter(torch.randn((n_embd, head_size)))

        #self.mask = torch.tril(torch.ones(context_size, context_size)) == 0
        #removendo máscara causal: não precisa pra classificação de série temporal

        torch.nn.init.kaiming_normal_(self.wq)
        torch.nn.init.kaiming_normal_(self.wk)
        torch.nn.init.kaiming_normal_(self.wv)

        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def self_attention_matricial(self, word_stack, wq, wk, wv, embedding_size):

      q = torch.matmul(word_stack, wq)
      k = torch.matmul(word_stack, wk)
      v = torch.matmul(word_stack, wv)
      scores = torch.matmul(q, torch.transpose(k, -2, -1)) / (embedding_size ** 0.5)

      #Mascara causal
      #scores = scores.masked_fill(self.mask, float('-inf'))

      probs = torch.softmax(scores, dim=-1)
      e_mtx = torch.matmul(probs, v)
      return e_mtx


    def forward(self, x):

      x = self.self_attention_matricial(x, self.wq, self.wk, self.wv, self.head_size) #B, L, D
      return x

class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """
  #cada cabeça projeta a atenção para um 'espaço' com suas matrizes key, querry, value

  def __init__(self, n_embd, num_heads, head_size, dropout):
      super().__init__()
      self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])
      self.proj = nn.Linear(n_embd, n_embd)
      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1) #B, L, D
      out = self.dropout(self.proj(out))
      return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 8 * n_embd),
            nn.ReLU(),
            nn.Linear(8 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

class Microtransformer(nn.Module):

    def __init__(self, sensor_size, context_size, num_classes, n_blocks, dropout, device):
        super().__init__()
        n_head = 1
        print(n_head)
        # each token directly reads off the logits for the next token from a lookup table
        self.position_embedding_table = nn.Embedding(context_size, sensor_size)
        self.blocks = nn.Sequential(*[Block(sensor_size, n_head=n_head, dropout=dropout) for _ in range(n_blocks)]) # * possibilita cada camada como argumento em sequential: (unpacking)
        self.ln_f = nn.LayerNorm(sensor_size) # final layer norm
        self.lm_head = nn.Linear(sensor_size * context_size, num_classes)
        self.device = device
       


    def forward(self, x):
        
        B, T, S = x.shape #Batch, time and sensors (float)

        pos_emb = self.position_embedding_table(torch.arange(T).to(self.device)) # (T, S) -> gets one embedding for each position
        
        
        x = x + pos_emb # (B, T, S)
        x = self.blocks(x) # (B, T, S)
        x = self.ln_f(x) # (B, T, S)
        x = x.view(B, -1) # (B, T * S)
        logits = self.lm_head(x) # (B, C)  # C: classes

        return logits



class HARTransformer(nn.Module):

    def __init__(self, input_shape, nhead, num_layers, num_classes, dropout = 0.15, sensor_group = 3):
        
        super(HARTransformer, self).__init__()

        self.batch, self.height, self.width = input_shape
        self.num_classes = num_classes

        for i in range(25, 1, -1):
            if self.height % i == 0:
                wordsize = i
                break

        dim_feedforward = self.height*self.width*2
        d_model = (self.width//sensor_group)*(self.height//wordsize)*22

        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.conv = nn.Sequential(
            self.make_conv_block(1, 22, kernel_size=(wordsize, sensor_group), stride=(wordsize, sensor_group)),
            nn.Flatten(), #batch, (w//sensor_group)*(height-2)
        )

        self.classification = nn.Linear(in_features=self.d_model, out_features = self.num_classes)

    def make_conv_block(self, input_channels, output_channels, kernel_size=3, stride = 1):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride),
            nn.LeakyReLU(0.2)
        )
    
    
    def forward(self, x):
        
        x = torch.unsqueeze(self.conv(torch.unsqueeze(x, 1)), 1)
        
        x = x.permute(1, 0, 2)
        
        # Multihead attention block
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Permute back to (batch_size, 1, n)
        x = x.permute(1, 0, 2)
        return torch.squeeze(self.classification(x), 1)
    

class CrossAttnHARTransformer(nn.Module):

    def __init__(self, input_shape, nhead, num_classes, dropout = 0.15, sensor_group = 3, wordsize = 64):
        
        super(CrossAttnHARTransformer, self).__init__()

        self.batch, self.height, self.width = input_shape
        self.num_classes = num_classes

        for i in range(12, 1, -1):
            if self.height % i == 0:
                kernelsize = i
                break

        dim_feedforward = self.height*self.width*2
        d_model = (self.width//sensor_group)*(self.height//kernelsize)//2

        self.d_model = d_model
        self.dropout_att = nn.Dropout(dropout)
        
        embed_dim = (self.width//sensor_group)*(self.height//kernelsize)//2
        
        self.norm_acc = nn.LayerNorm(embed_dim)
        self.norm_gyro = nn.LayerNorm(embed_dim)
        self.cross_att = nn.MultiheadAttention(embed_dim, nhead)
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim*wordsize)
        self.conv = nn.Sequential(
            self.make_conv_block(1, wordsize, kernel_size=(kernelsize, sensor_group), stride=(kernelsize, sensor_group)),
        )

        self._ff = self.make_ff_block(embed_dim*wordsize, embed_dim*wordsize*2, dropout=dropout)
        self._ff2 = self.make_ff_block(embed_dim*wordsize, embed_dim*wordsize*2, dropout=dropout)
        self.classification = nn.Linear(in_features=embed_dim*wordsize, out_features = self.num_classes)

    def make_ff_block(self, input_channels, output_channels, dropout=0.15):
        return nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(output_channels, input_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

    def make_conv_block(self, input_channels, output_channels, kernel_size=3, stride = 1):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride),
            nn.LeakyReLU(0.2)
        )
    
    
    def forward(self, x):
        
        x = self.conv(torch.unsqueeze(x, 1))
        
        x_acc = torch.flatten(x[:, :, :, ::2], start_dim = -2).permute(1, 0, 2)
        x_gyro = torch.flatten(x[:, :, :, 1::2], start_dim = -2).permute(1, 0, 2)
        
        x = self.cross_att(x_acc, x_gyro, x_gyro)[0]
        
        x = self.dropout_att(x)
        
        x = torch.flatten(x.permute(1, 0, 2), -2)
        
        x = self._ff(x)
        
        return torch.squeeze(self.classification(x), 1)

import math

class LoRAMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, *args, **kwargs):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=True, *args, **kwargs)
        self.has_adapted = False
        
    def adapt_rank(self, rank_percentage):
        
        if self.has_adapted:
            delta_in_proj = self.lora_B_in @ self.lora_A_in  # shape (3*embed_dim, embed_dim)
            self.in_proj_weight.data += delta_in_proj

            delta_out_proj = self.lora_B_out @ self.lora_A_out  # shape (embed_dim, embed_dim)
            self.out_proj.weight.data += delta_out_proj

        # Freeze original parameters
        self.in_proj_weight.requires_grad_(False)
        if self.in_proj_bias is not None:
            self.in_proj_bias.requires_grad_(False)
        self.out_proj.weight.requires_grad_(False)
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad_(False)
        
        self.has_adapted = True
        
        # Initialize LoRA parameters for in_proj (combined q, k, v)
        self.lora_A_in = nn.Parameter(torch.zeros(max(int(rank_percentage*self.embed_dim), 4), self.embed_dim))
        self.lora_B_in = nn.Parameter(torch.zeros(3 * self.embed_dim, max(int(rank_percentage*self.embed_dim), 4)))
        nn.init.normal_(self.lora_A_in, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B_in)

        # Initialize LoRA parameters for out_proj
        self.lora_A_out = nn.Parameter(torch.zeros(max(int(rank_percentage*self.embed_dim), 4), self.embed_dim))
        self.lora_B_out = nn.Parameter(torch.zeros(self.embed_dim, max(int(rank_percentage*self.embed_dim), 4)))
        nn.init.normal_(self.lora_A_out, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B_out)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, is_causal=False):

        if self.has_adapted:
            # Compute LoRA adjustments
            delta_in_proj = self.lora_B_in @ self.lora_A_in  # shape (3*embed_dim, embed_dim)
            adjusted_in_proj_weight = self.in_proj_weight + delta_in_proj

            delta_out_proj = self.lora_B_out @ self.lora_A_out  # shape (embed_dim, embed_dim)
            adjusted_out_proj_weight = self.out_proj.weight + delta_out_proj

        else:
            adjusted_in_proj_weight = self.in_proj_weight
            adjusted_out_proj_weight = self.out_proj.weight

        # Call the functional multi-head attention
        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            adjusted_in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            adjusted_out_proj_weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            is_causal=is_causal
        )
        return attn_output, attn_weights

def merge_AB(layer):
      def T(w):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w
      with torch.no_grad():
        layer.weight.data += T(layer.lora_B @ layer.lora_A)*layer.scaling
        nn.init.zeros_(layer.lora_B)

def copy_linear_to_lora(layer, percentage_rank):
      
      rank = max(int(min(layer.weight.shape[1], layer.weight.shape[0]) * (percentage_rank)), 4)
      new_layer = lora.Linear(layer.weight.shape[1], layer.weight.shape[0], r = rank).to(layer.weight.device)
      new_layer.weight = layer.weight
      new_layer.weight.requires_grad = False
      new_layer.bias = layer.bias
      nn.init.zeros_(new_layer.bias)
      new_layer.bias.requires_grad = False
        
      return new_layer

class LoraTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn = LoRAMultiheadAttention(embed_dim=kwargs["d_model"],num_heads= kwargs["nhead"], dropout=kwargs["dropout"])
        self.has_adapted = False

    def adapt_rank(self, percentage_rank):

        self.self_attn.adapt_rank(percentage_rank)

        if self.has_adapted:
            merge_AB(self.linear1)
            merge_AB(self.linear2)
        
        self.has_adapted = True
        self.new1 = copy_linear_to_lora(self.linear1, percentage_rank)
        self.new2 = copy_linear_to_lora(self.linear2, percentage_rank)

        delattr(self, 'linear1')
        delattr(self, 'linear2')

        self.linear1 = self.new1
        self.linear2 = self.new2

        delattr(self, 'new1')
        delattr(self, 'new2')

        gc.collect()
        torch.cuda.empty_cache()

class LoraHARTransformer(nn.Module):

    def __init__(self, input_shape, nhead, num_layers, num_classes, dropout = 0.15, sensor_group = 3, plas = 0.95):

        super(LoraHARTransformer, self).__init__()

        self.batch, self.height, self.width = input_shape
        self.num_classes = num_classes
        self.has_adapted = False
        self.current_step = 0
        self.plas = plas

        for i in range(25, 1, -1):
            if self.height % i == 0:
                wordsize = i
                break

        dim_feedforward = self.height*self.width*2
        d_model = (self.width//sensor_group)*(self.height//wordsize)*44

        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            LoraTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.conv = nn.Sequential(
            self.make_conv_block(1, 44, kernel_size=(wordsize, sensor_group), stride=(wordsize, sensor_group)),
            nn.Flatten(), #batch, (w//sensor_group)*(height-2)
        )

        self.classification = nn.Linear(in_features=self.d_model, out_features = self.num_classes)
        
    def make_conv_block(self, input_channels, output_channels, kernel_size=3, stride = 1):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride),
            nn.LeakyReLU(0.2)
        )

    '''    def adapt_rank(self, percentage_rank):

        if self.has_adapted:
            self.merge_AB(self.classification)
              
        self.has_adapted = True
        self.new1 = copy_linear_to_lora(self.classification, percentage_rank)
        
        delattr(self, 'classification')

        self.classification = self.new1

        delattr(self, 'new1')

        gc.collect()
        torch.cuda.empty_cache()'''

    def calculate_percentage_rank(self):
        self.current_step += 1
        return 1/2 + (1/2*(self.plas**self.current_step))

    def experience_end(self):
      percentage = self.calculate_percentage_rank()

      for layer in self.layers:
        layer.adapt_rank(percentage)

    def forward(self, x):

        x = torch.unsqueeze(self.conv(torch.unsqueeze(x, 1)), 1)

        x = x.permute(1, 0, 2)

        # Multihead attention block
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Permute back to (batch_size, 1, n)
        x = x.permute(1, 0, 2)
        return torch.squeeze(self.classification(x), 1)