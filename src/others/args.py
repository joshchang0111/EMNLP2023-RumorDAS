import os
import ipdb
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from transformers import Seq2SeqTrainingArguments #TrainingArguments
from transformers.trainer_utils import IntervalStrategy

def args_post_init(model_args, data_args, training_args):
	"""Post process of arguments after parsing"""
	model_args.model_name = model_args.model_name_or_path
	training_args.output_root = training_args.output_dir

	## Setup wandb reporter
	if "wandb" in training_args.report_to:
		training_args.run_name = "{}/{}/fold-{}".format(training_args.exp_name, data_args.dataset_name, data_args.fold)
	
	if training_args.task_type == "train_detector":
		
		## Set results path
		training_args.overall_results_path = "{}/{}/{}/overall_results.txt".format(training_args.output_dir, data_args.dataset_name, training_args.exp_name)
		training_args.output_dir = "{}/{}/{}/{}".format(training_args.output_dir, data_args.dataset_name, training_args.exp_name, data_args.fold)

		if model_args.summarizer_name_or_path is not None:
			if model_args.summarizer_name_or_path != "use_tuned_summarizer":
				training_args.output_dir = "{}+summ".format(training_args.output_dir)
			else:
				training_args.output_dir = "{}+tuned_summ".format(training_args.output_dir)
		os.makedirs(training_args.output_dir, exist_ok=True)

		## Initialize `overall_results.txt`
		if not os.path.isfile(training_args.overall_results_path):
			with open(training_args.overall_results_path, "w") as fw:
				f1s = ["{:10s}".format("F1-{}".format(label_i)) for label_i in range(data_args.num_labels)]
				metrics2report = [
					"{:20s}".format("Model"), 
					"{:10s}".format("Summarizer"),
					"{:8s}".format("Mode"),
					"{:20s}".format("Fold"),
					"{:10s}".format("Acc"),
					"{:10s}".format("F1-Macro")
				]
				metrics2report.extend(f1s)
				fw.write("{}\n".format("\t".join(metrics2report)))
	
	elif training_args.task_type == "train_adv_stage1" or \
		 training_args.task_type == "train_adv_stage2":
		
		training_args.overall_results_path = "{}/{}/{}/overall_results.txt".format(training_args.output_dir, data_args.dataset_name, training_args.exp_name)
		training_args.output_dir = "{}/{}/{}/{}".format(training_args.output_dir, data_args.dataset_name, training_args.exp_name, data_args.fold)
		os.makedirs(training_args.output_dir, exist_ok=True)

		## Initialize `overall_results.txt`
		if not os.path.isfile(training_args.overall_results_path):
			with open(training_args.overall_results_path, "w") as fw:
				f1s = ["{:10s}".format("F1-{}".format(label_i)) for label_i in range(data_args.num_labels)]
				metrics2report = [
					"{:8s}".format("Mode"), ## train / evaluate 
					"{:10s}".format("attack"), ## v / -
					"{:10s}".format("summarizer"), ## v / -
					"{:20s}".format("Fold"),
					"{:10s}".format("Acc"),
					"{:10s}".format("F1-Macro")
				]
				metrics2report.extend(f1s)
				metrics2report.append("{:10s}".format("ROUGE-1"))
				metrics2report.append("{:10s}".format("ROUGE-2"))
				metrics2report.append("{:10s}".format("ROUGE-L"))
				fw.write("{}\n".format("\t".join(metrics2report)))

	elif training_args.task_type == "ssra_loo" or \
		 training_args.task_type == "ssra_kmeans":

		training_args.overall_results_path = "{}/{}/{}/overall_results.txt".format(training_args.output_dir, data_args.dataset_name, training_args.exp_name)
		training_args.output_dir = "{}/{}/{}/{}".format(training_args.output_dir, data_args.dataset_name, training_args.exp_name, data_args.fold)
		os.makedirs(training_args.output_dir, exist_ok=True)

		## Initialize `overall_results.txt`
		if not os.path.isfile(training_args.overall_results_path):
			with open(training_args.overall_results_path, "w") as fw:
				metrics2report = ["{:20s}".format("Fold")]
				metrics2report.append("{:10s}".format("ROUGE-1"))
				metrics2report.append("{:10s}".format("ROUGE-2"))
				metrics2report.append("{:10s}".format("ROUGE-L"))
				fw.write("{}\n".format("\t".join(metrics2report)))

	elif training_args.task_type == "train_filter":
		training_args.output_dir = "{}/{}/{}/{}".format(training_args.output_dir, data_args.dataset_name, training_args.exp_name, data_args.fold)
		os.makedirs(training_args.output_dir, exist_ok=True)

	elif training_args.task_type == "build_cluster_summary":
		print("", end="")

	else:
		raise ValueError("training_args.task_type not specified!")

	## Experiment
	if training_args.framing and training_args.do_eval and not training_args.do_train:
		## Framing Effect
		training_args.framing_stats_file = "{}/{}/{}/framing_stats.txt".format(training_args.output_root, data_args.dataset_name, training_args.exp_name)
		if not os.path.isfile(training_args.framing_stats_file):
			with open(training_args.framing_stats_file, "w") as fw:
				fw.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
					"Fold", 
					"Type",
					"Pred Diff>",
					"# fra_resp", 
					"# pos_resp", ## positive framing responses
					"# neg_resp", ## negative framing responses
					"# tot_resp", 
					"# fra_post", 
					"# pos_post", ## positive framed posts
					"# neg_post", ## negative framed posts
					"# tot_post"
				))

	elif training_args.explain:
		## Explainabilty
		training_args.overall_results_path = "{}/{}/{}/explain_results.txt".format(training_args.output_root, data_args.dataset_name, training_args.exp_name)
		if not os.path.isfile(training_args.overall_results_path):
			with open(training_args.overall_results_path, "w") as fw:
				f1s = ["{:10s}".format("F1-{}".format(label_i)) for label_i in range(data_args.num_labels)]
				metrics2report = [
					"{:8s}".format("Mode"), ## train / evaluate 
					"{:10s}".format("attack"), ## v / -
					"{:10s}".format("summarizer"), ## v / -
					"{:20s}".format("Fold"),
					"{:10s}".format("Acc"),
					"{:10s}".format("F1-Macro")
				]
				metrics2report.extend(f1s)
				metrics2report.append("{:10s}".format("ROUGE-1"))
				metrics2report.append("{:10s}".format("ROUGE-2"))
				metrics2report.append("{:10s}".format("ROUGE-L"))
				fw.write("{}\n".format("\t".join(metrics2report)))

	return model_args, data_args, training_args

@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
	"""Customized TrainingArguments"""

	## Overwrite TrainingArguments default value
	output_dir: str = field(
		default="/mnt/1T/projects/RumorV2/results",
		metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
	)
	num_train_epochs: float = field(
		default=10.0, 
		metadata={"help": "Total number of training epochs to perform."},
	)
	evaluation_strategy: IntervalStrategy = field(
		default="epoch",
		metadata={"help": "The evaluation strategy to use."},
	)
	save_strategy: IntervalStrategy = field(
		default="epoch",
		metadata={"help": "The checkpoint save strategy to use."},
	)
	save_total_limit: Optional[int] = field(
		default=1,
		metadata={
			"help": (
				"Limit the total amount of checkpoints. "
				"Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
			)
		},
	)
	overwrite_output_dir: bool = field(
		default=True,
		metadata={
			"help": (
				"Overwrite the content of the output directory. "
				"Use this to continue training if output_dir points to a checkpoint directory."
			)
		},
	)
	predict_with_generate: bool = field(
		default=True, 
		metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
	)
	report_to: Optional[str] = field(
		default="none", 
		metadata={"help": "The list of integrations to report the results and logs to."}
	)

	##################
	## Self-defined ##
	##################
	task_type: Optional[str] = field(
		default=None,
		metadata={
			"help": "Which task to do:"
			"train_detector, train_adv_stage1, train_adv_stage2" ## Train detector / attacker (generator)
			"train_filter, build_cluster_summary, ssra_loo, ssra_kmeans" ## Train summarizer
		}
	)
	attack_type: Optional[str] = field(
		default="untargeted",
		metadata={"help": "Which type of attack when train_adv_stage2: [shift, reverse, untargeted]"}
	)
	save_model_accord_to_metric: bool = field(
		default=True,
		metadata={"help": "Whether to save model according to metric (f1-macro / accuracy) instead of loss."},
	)
	overall_results_path: Optional[str] = field(
		default=None,
		metadata={"help": "File path of overall results."}
	)
	best_checkpoint_path: Optional[str] = field(
		default=None, 
		metadata={"help": "Path of the directory that contains checkpoint you want to load."}
	)
	exp_name: Optional[str] = field(
		default=None, 
		metadata={"help": "Experiment name, also the name of output folder."}
	)
	output_root: Optional[str] = field(
		default=None,
		metadata={"help": "Record for later usage."}
	)

	## Options for `exp.py`
	framing: bool = field(
		default=False, 
		metadata={"help": "Whether to perform experiments to observe framing effect."}
	)
	explain: bool = field(
		default=False, 
		metadata={"help": "Whether to perform experiments about explainability, use summarizer without attack"}
	)
	pure_analysis: bool = field(
		default=False, 
		metadata={"help": "Whether to ignore building datasets, model and trainer for faster analysis."}
	)

@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.

	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""

	#############################
	## For text classification ##
	#############################
	task_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the task to train on."},
	)
	dataset_name: Optional[str] = field(
		default=None, 
		metadata={"help": "The name of the dataset to use (via the datasets library)."}
	)
	dataset_config_name: Optional[str] = field(
		default=None, 
		metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	max_seq_length: int = field(
		default=32, #128,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	overwrite_cache: bool = field(
		default=False, 
		metadata={"help": "Overwrite the cached preprocessed datasets or not."}
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
			"value if set."
		},
	)
	train_file: Optional[str] = field(
		default=None, 
		metadata={"help": "A csv or a json file containing the training data."}
	)
	validation_file: Optional[str] = field(
		default=None, 
		metadata={"help": "A csv or a json file containing the validation data."}
	)
	test_file: Optional[str] = field(
		default=None, 
		metadata={"help": "A csv or a json file containing the test data."}
	)

	#######################
	## For summarization ##
	#######################
	lang: str = field(default=None, metadata={"help": "Language id for summarization."})
	text_column: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
	)
	summary_column: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
	)
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	max_source_length: Optional[int] = field(
		default=1024,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	max_target_length: Optional[int] = field(
		default=128,
		metadata={
			"help": "The maximum total sequence length for target text after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	val_max_target_length: Optional[int] = field(
		default=None,
		metadata={
			"help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
			"This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
			"during ``evaluate`` and ``predict``."
		},
	)
	num_beams: Optional[int] = field(
		default=None,
		metadata={
			"help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
			"which is used during ``evaluate`` and ``predict``."
		},
	)
	ignore_pad_token_for_loss: bool = field(
		default=True,
		metadata={
			"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
		},
	)
	source_prefix: Optional[str] = field(
		default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
	)

	forced_bos_token: Optional[str] = field(
		default=None,
		metadata={
			"help": "The token to force as the first generated token after the decoder_start_token_id."
			"Useful for multilingual models like mBART where the first generated token"
			"needs to be the target language token (Usually it is the target language token)"
		},
	)

	##################
	## Self-defined ##
	##################
	dataset_root: Optional[str] = field(
		default="../dataset", 
		metadata={"help": "Root path of all datasets."}
	)
	fold: Optional[str] = field(
		default=None, 
		metadata={"help": "Which fold of dataset to used for training."}
	)
	max_tree_length: Optional[int] = field(
		default=15, 
		metadata={
			"help": 
				"Maximum number of nodes (including source) in each tree -> len(source + replies)"
				"Should leave one more node for generation"
		}
	)
	num_labels: Optional[int] = field(
		default=None, 
		metadata={"help": "Number of labels for classification task."}
	)
	summ_model: str = field(
		default="t5-small", 
		metadata={"help": "Model used to predict stance summary."}
	)

	content_file: Optional[str] = field(
		default="data.csv", 
		metadata={"help": "Content of dataset."}
	)
	max_tweet_length: Optional[int] = field(
		default=32,
		metadata={
			"help": "The maximum length of each tweet after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	min_target_length: Optional[int] = field(
		default=None, 
		metadata={"help": "minimum generation length for SSRA-LOO"}
	)

	## *** Input type ***
	split_3_groups: bool = field(
		default=False, 
		metadata={"help": "Whether to random split each tree into three groups for ssra_loo."}
	)
	augmentation: bool = field(
		default=False, 
		metadata={"help": "Whether to use data augmentation (sequential subtrees) or not."}
	)
	summary_type: Optional[str] = field(
		default="no",
		metadata={"help": "no / src_only / stance / all1 / all4"}
	)

	## *** Experiments: framing-response effect ***
	#framing_response_mode: Optional[str] = field(
	#	default=None, 
	#	metadata={"help": "train / test"}
	#)

	## Expeirment: for comparing with GACL
	gacl: bool = field(
		default=False, 
		metadata={"help": "Whether to train the model on the dataset of GACL"}
	)
	gacl_path: Optional[str] = field(
		default=None, 
		metadata={"help": "Path to GACL dataset"}
	)

	def __post_init__(self):
		if self.val_max_target_length is None:
			self.val_max_target_length = self.max_target_length

		## Adjust num_labels to different datasets
		self.num_labels = pd.read_csv("{}/{}/data.csv".format(self.dataset_root, self.dataset_name))["veracity"].nunique()

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""
	##############################
	## From text classification ##
	##############################
	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, 
		metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, 
		metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)

	#################################
	## From Seq2Seq ModelArguments ##
	#################################
	resize_position_embeddings: Optional[bool] = field(
		default=None,
		metadata={
			"help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
			"the model's position embeddings."
		},
	)

	##################
	## Self-defined ##
	##################
	model_name: str = field(
		default=None, 
		metadata={"help": "Backbone (main) model name, used for recording results."}
	)

	## For summarizer
	abstractor_name_or_path: str = field(
		default=None, 
		metadata={
			"help": "Path to pretrained `summarizer` or model identifier from huggingface.co/models"
			"only need to specify when using adv. training."
		}
	)
	extractor_name_or_path: str = field(
		default=None, 
		metadata={
			"help": "Pre-trained `extractor` if needed, otherwise specify the extractor type"
			"random / kmeans / autoencoder / multi_ae"
		}
	)
	summarizer_output_type: str = field(
		default=None, 
		metadata={
			"help": "a string that describes the summarization output type in the function `extract_then_abstract`."
		}
	)
	summarizer_name_or_path: str = field(
		default=None, 
		metadata={
			"help": "Path to pretrained `summarizer` or model identifier from huggingface.co/models"
			"only need to specify when using adv. training."
		}
	)
	cluster_type: str = field(
		default=None, 
		metadata={
			"help": "kmeans / topics"
		}
	)
	num_clusters: int = field(
		default=None
	)
	cluster_mode: str = field(
		default=None, 
		metadata={
			"help": "either `train` or `test`"
		}
	)
	extract_type: str = field(
		default=None, 
		metadata={
			"help": "random / k-means / autoencoder"
		}
	)
	extract_ratio: float = field(
		default=None, 
		metadata={
			"help": "Extract ratio for extractive response summarization"
		}
	)
	filter_ratio: float = field(
		default=None, 
		metadata={
			"help": "Filter ratio for response filter (autoencoder)"
		}
	)
	filter_layer_enc: int = field(default=2)
	filter_layer_dec: int = field(default=2)
	target_class_ext_ae: str = field(
		default="all", 
		metadata={
			"help": "Used for train_extract_ae, the target class of responses we desire the autoencoder to reconstruct"
		}
	)
	use_mc_dropout_for_summarizer: Optional[bool] = field(
		default=False,
		metadata={
			"help": "Monte-Carlo dropout for response summarizer"
		}
	)

	## For detector head (GCN)
	edge_filter: Optional[bool] = field(
		default=False, 
		metadata={
			"help": "Whether to add edge filter to GCN"
		}
	)
	td_gcn: Optional[bool] = field(
		default=False, 
		metadata={
			"help": "Whether to use top-down GCN or not"
		}
	)
	bu_gcn: Optional[bool] = field(
		default=False, 
		metadata={
			"help": "Whether to use bottom-up GCN or not"
		}
	)
	root_enhance: Optional[bool] = field(
		default=False, 
		metadata={
			"help": "Whether to use root feature enhancement"
		}
	)

	def __post_init__(self):
		self.model_name = self.model_name_or_path

