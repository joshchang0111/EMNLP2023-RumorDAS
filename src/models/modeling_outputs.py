from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.file_utils import ModelOutput

@dataclass
class BaseModelOutputWithEmbedding(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    ## NEW: return embeddings
    embed_tok: Optional[torch.FloatTensor] = None ## Token embeddings
    embed_pos: Optional[torch.FloatTensor] = None ## Position embeddings
    embed_out: Optional[torch.FloatTensor] = None ## Total embeddings = (Token embeddings) + (Position embeddings)
    #outputs_embeds: Optional[torch.FloatTensor] = None ## Total embeddings = (Token embeddings) + (Position embeddings)

@dataclass
class Seq2SeqWithSequenceClassifierOutput(ModelOutput):
	"""
	Output for seq2seq models along with sequence classifier.
	
	Seq2Seq (generation):
		- loss
		- logits
		- past_key_values
		- decoder_hidden_states
		- decoder_attentions
		- cross_attentions
		- encoder_last_hidden_state
		- encoder_hidden_states
		- encoder_attentions

	Sequence Classifier:
		- loss_det
		- logits_det
	"""

	## For Seq2Seq
	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

	## For classifier
	loss_det: Optional[torch.FloatTensor] = None
	logits_det: torch.FloatTensor = None
	#hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	#attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class RumorDetectorOutput(ModelOutput):
	"""
	Output for Rumor Detector + Response Generator + Response Summarizer.
	
	Response Generator:
		- loss
		- logits
		- past_key_values
		- decoder_hidden_states
		- decoder_attentions
		- cross_attentions
		- encoder_last_hidden_state
		- encoder_hidden_states
		- encoder_attentions

	Rumor Detector:
		- loss_det
		- logits_det

	Response Summarizer:
		- summary_tokens: for abstractive summarizer (abstractor)
		- n_extracted_attack: for extractive summarizer (extractor)
	"""

	## Response Generator
	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

	## Rumor Detector
	loss_det: Optional[torch.FloatTensor] = None
	logits_det: torch.FloatTensor = None
	
	## Response Summarizer
	summary_tokens: Optional[torch.FloatTensor] = None
	n_ext_adv: int = None
	filt_ext_idxs: Optional[torch.Tensor] = None
	clus_ext_idxs: Optional[torch.Tensor] = None
	clus_ids: Optional[torch.Tensor] = None
