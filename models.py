import torch
import torch.nn as nn
from transformers import AutoModel
import config

class ModelPart1(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.BASE_MODEL_PATH, local_files_only=True)
        bert_output_dim = 768
        self.bottleneck = nn.Linear(bert_output_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, config.NUM_LABELS)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = bert_output.last_hidden_state[:, 0, :]
        bottleneck_output = self.bottleneck(cls_output)
        logits = self.classifier(self.activation(bottleneck_output))
        return logits


class ModelPart2(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.BASE_MODEL_PATH, local_files_only=True)
        bert_output_dim = 768
        intermediate_dim = 256 # just a hyperparameter intermediate as reducing to directly Y=[8,32,64,128] would be too much.
        self.encoder = nn.Sequential(
            nn.Linear(bert_output_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, bert_output_dim)
        )
        self.classifier = nn.Linear(bert_output_dim, config.NUM_LABELS)#chnaged it to bert_output_dim

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_original = bert_output.last_hidden_state[:, 0, :]
        Y = self.encoder(cls_original)
        reconstructed_cls = self.decoder(Y)
        logits = self.classifier(reconstructed_cls)# changed it
        return logits, reconstructed_cls, cls_original


class ModelPart3(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.BASE_MODEL_PATH, local_files_only=True)
        bert_output_dim = 768
        intermediate_dim = 256 # just a hyperparameter intermediate as reducing to directly Y=[8,32,64,128] would be too much.
        self.encoder = nn.Sequential(
            nn.Linear(bert_output_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, bert_output_dim)
        )
        self.classifier = nn.Linear(bert_output_dim, config.NUM_LABELS)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_original = bert_output.last_hidden_state[:, 0, :]
        Y = self.encoder(cls_original)
        
        bottleneck_for_downstream = Y
        
        if self.training:
            noise = torch.randn_like(Y) * config.NOISE_LEVEL
            bottleneck_for_downstream = Y + noise
        
        reconstructed_cls = self.decoder(bottleneck_for_downstream)
        logits = self.classifier(reconstructed_cls)
        
        return logits, reconstructed_cls, cls_original