import torch.nn.functional as F
import torch
from torch import nn
from ..SubNets.transformers_encoder.transformer import TransformerEncoder

class Enhance_LLM(nn.Module):
    def __init__(self, args):
        super(Enhance_LLM, self).__init__()
        self.args = args
        self.dst_feature_dims = args.enhance_dst_feature_dims

        self.num_labels = args.num_labels
        
        self.interact_layers = args.enhance_feat_fusion_interact_layers
        self.interact_num_heads = args.enhance_feat_fusion_interact_num_heads
        self.attn_dropout = args.enhance_feat_fusion_attn_dropout
        self.relu_dropout = args.enhance_feat_fusion_relu_dropout
        self.embed_dropout = args.enhance_feat_fusion_embed_dropout
        self.res_dropout = args.enhance_feat_fusion_res_dropout
        self.attn_mask = args.enhance_feat_fusion_attn_mask


        self.encoders = self.get_transformer_encoder(self.dst_feature_dims, self.interact_layers)


        self.classifier = nn.Linear(self.dst_feature_dims, self.num_labels)
        self.dropout = nn.Dropout(self.args.enhance_classifier_dropout if hasattr(args, 'enhance_classifier_dropout') else 0.1)


    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.interact_num_heads,
            layers=layers,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )

    def forward(self, textual):
        """
        Args:
            textual: tensor of shape (batch_size, seq_len, feature_dim)
                     e.g., (16, 128, 768)
        Returns:
            output: dict containing:
                - 'logits': (B, num_labels)
                - 'all_layer_sequences': list of Tensors, each (B, L, D), len = num_layers + 1 (包括输入)
                - 'all_layer_pooled': list of Tensors, each (B, D), len = num_layers + 1
        """
        x = textual  # (B, L, D), already from BERT

        batch_size = x.size(0)

        all_layer_sequences = []
        all_layer_pooled = []

        cls_pooled = x[:, 0, :]  # [CLS] pooling
        # 或者使用 mean pooling
        # cls_pooled = x.mean(dim=1)
        all_layer_sequences.append(x)  # (B, L, D)
        all_layer_pooled.append(cls_pooled)  # (B, D)

        x_transposed = x.transpose(0, 1)  # (L, B, D)

        current_input = x_transposed
        for layer_idx, layer_module in enumerate(self.encoders.layers):

            current_input = layer_module(current_input)  # (L, B, D)

            layer_output = current_input.transpose(0, 1)  # (B, L, D)

            all_layer_sequences.append(layer_output)


            cls_pooled = layer_output[:, 0, :]  # (B, D)
            all_layer_pooled.append(cls_pooled)

        final_pooled = all_layer_pooled[-1]  # (B, D)
        final_pooled = self.dropout(final_pooled)
        logits = self.classifier(final_pooled)  # (B, num_labels)

        return {
            'logits': logits,
            'all_layer_sequences': all_layer_sequences,        # list of (B, L, D), len = layers + 1
            'all_layer_pooled': all_layer_pooled,              # list of (B, D), len = layers + 1
        }

class Enhance_Models(nn.Module):
    def __init__(self, main_dim, aux_dim, output_dim, dropout):
        super(Enhance_Models, self).__init__()
        self.main_dim = main_dim
        self.aux_dim = aux_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fusion = nn.Sequential(
            nn.Linear(self.main_dim + self.aux_dim, self.main_dim + self.aux_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.main_dim + self.aux_dim, self.main_dim + self.aux_dim),
            nn.LayerNorm(self.main_dim + self.aux_dim)  # 加 LayerNorm 提高稳定性
        )
        self.downsample = nn.Linear(self.main_dim, self.output_dim) if self.main_dim != self.output_dim else None

    def forward(self, h, e):
        """
        h: (B, main_dim)
        e: (B, aux_dim)
        """
        residual = self.downsample(h) if self.downsample is not None else h
        fused = torch.cat([h, e], dim=-1)  # (B, main_dim + aux_dim)
        h_out = self.fusion(fused)
        h_out = h_out + residual  # 残差连接
        return h_out

class ProgressiveEnhancer(nn.Module):
    def __init__(self, args):
        super(ProgressiveEnhancer, self).__init__()
        self.main_dim = args.enhance_main_dim
        self.aux_dim = args.enhance_aux_dim
        self.output_dim = args.enhance_main_dim+args.enhance_aux_dim  # 融合后维度
        self.num_layers = args.enhance_feat_fusion_interact_layers     
        self.dropout = args.enhance_dropout

        self.blocks = nn.ModuleList([
            Enhance_Models(
                main_dim=self.main_dim if i == 0 else self.output_dim,
                aux_dim=self.aux_dim,
                output_dim=self.output_dim,
                dropout=self.dropout
            )
            for i in range(self.num_layers)
        ])

    def forward(self, h_main, all_layer_pooled):
        """
        h_main: (B, 1536)
        all_layer_pooled: list of Tensors, each (B, 768), len = L >= num_layers
        """
        h = h_main

        available_layers = min(self.num_layers, len(all_layer_pooled)-1)
        for i in range(available_layers):
            e = all_layer_pooled[i+1]
            h = self.blocks[i](h, e)
        return h 



class ProgressiveEnhancerWithAuxToken(nn.Module):
    def __init__(self, args):
        super(ProgressiveEnhancerWithAuxToken, self).__init__()
        self.hidden_dim = args.enhance_dst_feature_dims
        self.num_fusion_layers = args.enhance_feat_fusion_interact_layers

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model= self.hidden_dim,
            nhead=8,
            dim_feedforward= self.hidden_dim * 4,
            dropout= args.enhance_transformer_dropout,
        )

        if  args.num_transformer_layers_per_fusion > 1:
            self.inner_transformer = nn.TransformerEncoder(
                encoder_layer=self.transformer_layer,
                num_layers= args.num_transformer_layers_per_fusion,
                norm=nn.LayerNorm(self.hidden_dim)
            )
        else:
            self.inner_transformer = None


    def forward(self, X_main, all_layer_pooled):
        """
        """
        B, L_main, D = X_main.shape

        e_list = all_layer_pooled[1:1 + self.num_fusion_layers]  # 取前 L 个中间层输出

        e1 = e_list[0]  # (B, 768)
        e1_expanded = e1.unsqueeze(1)
        X = torch.cat([X_main, e1_expanded], dim=1)

        for i, e in enumerate(e_list):
            if i == 0:
                pass
            else:
                X[:, -1, :] = e  # (B, 768)

            if self.inner_transformer is not None:
                X = self.inner_transformer(X)
            else:
                X = self.transformer_layer(X)

        return X
