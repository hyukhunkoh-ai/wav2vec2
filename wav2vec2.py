from torch import nn, Tensor
from typing import Optional, Tuple
from components import _get_feature_extractor, _get_encoder, FeatureProjection
from vector_quantizer import Wav2Vec2GumbelVectorQuantizer
from compute_mask_idx import _compute_mask_indices

import torch

################
codevector_dim = 256
context_size = 768
proj_codevector_dim = 256
in_features, embed_dim, dropout_input = 512, 768, 0.1
featureprojection = FeatureProjection(in_features, embed_dim, dropout_input)
mask_time_length = 10
mask_time_prob = 0.065
num_codevectors_per_group = 320
num_codevector_groups = 2
diversity_loss_weight = 0.1
initializer_range = 0.2
num_negatives = 100
contrastive_logits_temperature = 0.1
#############


extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
feature_extractor = _get_feature_extractor(
        'group_norm', extractor_conv_layer_config, False)
encoder = _get_encoder(
    in_features=extractor_conv_layer_config[-1][0],
    embed_dim=768,
    dropout_input=0.1,
    pos_conv_kernel=128,
    pos_conv_groups=16,
    num_layers=12,
    num_heads=12,
    attention_dropout=0.1,
    ff_interm_features=3072,
    ff_interm_dropout=0.1,
    dropout=0.1,
    layer_norm_first=False,
    layer_drop=0.05,
    num_out=768,
)


class Wav2Vec2Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_projection = featureprojection
        self.encoder = encoder
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())


        ### pretrain###
        self.quantizer = Wav2Vec2GumbelVectorQuantizer()
        self.project_q = nn.Linear(codevector_dim, proj_codevector_dim)#from codebook to compare
        self.project_hid = nn.Linear(context_size, proj_codevector_dim) # from c to compare

        self.apply(self._init_weights)

    def extract_features(self,waveforms: Tensor,
                         lengths: Optional[Tensor] = None,) -> Tuple[Tensor, Optional[Tensor]]:
        return self.feature_extractor(waveforms, lengths)

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):


        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=mask_time_prob,
                mask_length=mask_time_length,
                device=hidden_states.device,
                attention_mask=attention_mask,
                min_masks=2,
            )
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        return hidden_states,mask_time_indices

    def set_gumbel_temperature(self, temperature: int):
        return self.quantizer.set_temperature(temperature)

    def _init_weights(self, module):
        if isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight.data)

        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def _sample_negatives(
        features: torch.FloatTensor, num_negatives: int, attention_mask: Optional[torch.LongTensor] = None
    ):
        batch_size, sequence_length, hidden_size = features.shape
        if sequence_length <= 1:
            raise ValueError(
                f"`features should have `sequence_length` > 1, but are of shape (batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
            )

        features = features.view(-1, hidden_size)  # B,l,C => (B*l),C

        with torch.no_grad():

            sampled_negative_indices = []
            for batch_idx in range(batch_size):
                high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
                sampled_indices_slice = torch.randint(
                    0, high, size=(num_negatives * sequence_length,), device=features.device
                )
                sampled_negative_indices.append(sampled_indices_slice)

            sampled_negative_indices = torch.stack(sampled_negative_indices)



            feature_indices = (
                torch.arange(sequence_length, device=features.device)[:, None]
                .expand(sequence_length, num_negatives)
                .flatten()
            )


            sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

        for batch_idx in range(1, batch_size):
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        sampled_negatives = features[sampled_negative_indices.view(-1)]
        sampled_negatives = sampled_negatives.view(batch_size, sequence_length, num_negatives, hidden_size).permute(
            2, 0, 1, 3
        )

        return sampled_negatives     # K,b,l,256

    @staticmethod
    def compute_contrastive_logits(
            target_features: torch.FloatTensor, # 1,b,l,256
            negative_features: torch.FloatTensor,
            predicted_features: torch.FloatTensor,  # b,l,256
            temperature=1.0,
    ):
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        logits = logits / temperature # 첫번째는 유사도가 높아야함
        return logits

    def calculate_loss(self,waveforms: Tensor, lengths: Optional[Tensor] = None,) -> Tensor:

        extract_x, lengths = self.feature_extractor(waveforms, lengths)
        transformer_x = self.feature_projection(extract_x)

        hidden_states, mask_time_indices = self._mask_hidden_states(transformer_x)
        encoder_outputs = self.encoder(hidden_states,lengths)

        transformer_features = self.project_hid(encoder_outputs)

        quantized_features, codevector_perplexity = self.quantizer(extract_x, mask_time_indices)
        quantized_features = self.project_q(quantized_features) # z->q(b,l,256)


        negative_quantized_features = self._sample_negatives(
            quantized_features, num_negatives, attention_mask=None
        )

        logits = self.compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            transformer_features,
            0.1)

        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf") # k,b,l

        preds = logits.transpose(0, 2).reshape(-1, logits.size(0))

        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
        contrastive_loss = nn.functional.cross_entropy(preds.float(), target, reduction="sum")

        num_codevectors = num_codevectors_per_group * num_codevector_groups
        diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors

        loss = contrastive_loss + diversity_loss_weight * diversity_loss

        return loss

    def forward(self,x,length=None):

        l, length = self.feature_extractor(x,length)
        l = self.feature_projection(l)
        l = self.encoder(l, length)

        return l

    def device_to(self,device):
        self.feature_extractor.to(device)
        self.feature_projection.to(device)
        self.encoder.to(device)
        self.masked_spec_embed.to(device)

        self.quantizer.to(device)
        self.project_q.to(device)
        self.project_hid.to(device)


