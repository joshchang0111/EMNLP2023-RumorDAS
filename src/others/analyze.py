import ipdb
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Analyze experiment results")

	## Which experiment
	parser.add_argument("--framing_response", action="store_true")

	## Others
	parser.add_argument("--dataset", type=str, default="semeval2019", choices=["semeval2019", "Pheme", "twitter15", "twitter16"])
	parser.add_argument("--data_root", type=str, default="../dataset/processed")
	parser.add_argument("--data_root_V2", type=str, default="../dataset/processedV2")
	parser.add_argument("--fold", type=str, default="0,1,2,3,4", help="either use 5-fold data or train/dev/test from rumoureval2019 competition")
	parser.add_argument("--result_path", type=str, default="/mnt/hdd1/projects/RumorV2/results")

	args = parser.parse_args()

	return args

def framing_response(args):
	def read_predict_results(file):
		result_dict = {}
		with open(file) as reader:
			for idx, line in enumerate(reader.readlines()):
				if idx == 0:
					continue
				line = line.strip().rstrip().replace(" ", "")
				tree_id, pred, gt = line.split("\t")
				src_id , subtree_id = tree_id.split("_")
				
				if src_id not in result_dict:
					result_dict[src_id] = {}
				result_dict[src_id][int(subtree_id)] = {}
				result_dict[src_id][int(subtree_id)]["prediction"] = int(pred)
				result_dict[src_id][int(subtree_id)]["gt_label"] = int(gt)

		return result_dict

	def fsr(result_dict):
		total_response, framing_response = 0, 0
		pos_framing, neg_framing = 0, 0
		pos_framing_list = []
		neg_framing_list = []
		for src_id in result_dict.keys():
			impact_response = None ## Default to positive impact
			for subtree_id in result_dict[src_id].keys():
				pred = result_dict[src_id][subtree_id]["prediction"]
				gt = result_dict[src_id][subtree_id]["gt_label"]
		
				if subtree_id == 1: ## Source only
					if pred == gt:
						impact_src = 1
					else:
						impact_src = 0
				else: ## Source + response
					if pred == gt:
						impact_response = 1
					else:
						impact_response = 0
					impact_response = impact_response - impact_src
					
					total_response = total_response + 1
					if impact_response != 0:
						framing_response = framing_response + 1
						if impact_response > 0:
							pos_framing = pos_framing + 1
							pos_framing_list.append("{}_{}".format(src_id, subtree_id))
						else:
							neg_framing = neg_framing + 1
							neg_framing_list.append("{}_{}".format(src_id, subtree_id))
		
					impact_src = impact_response
				#print("subtree_id: {:2d}, prediction: {}, gt_label: {}, impact_response: {}".format(subtree_id, pred, gt, impact_response))
		
		print("# total   responses: {}".format(total_response))
		print("# framing responses: {}".format(framing_response))
		print("Framing Success Rate (FSR): {:.2f}%".format(framing_response * 100 / total_response))
		print("Positive Framing: {} ({:.2f}%)".format(pos_framing, pos_framing * 100 / total_response))
		print("Negative Framing: {} ({:.2f}%)".format(neg_framing, neg_framing * 100 / total_response))

		return pos_framing_list, neg_framing_list

	print("Analyzing framing-response experiments")
	print("[Dataset]: {}".format(args.dataset))
	print("[Fold]   : {}".format(args.fold))

	predict_file_raw = "{}/{}/framing-response/{}/predict_results_framing_responses.txt".format(args.result_path, args.dataset, args.fold)
	predict_file_sum = "{}/{}/framing-response-summ/{}/predict_results_framing_responses.txt".format(args.result_path, args.dataset, args.fold)
	result_raw = read_predict_results(predict_file_raw)
	result_sum = read_predict_results(predict_file_sum)

	'''
	raw_ids = result_raw.keys()
	sum_ids = result_sum.keys()
	for raw_id in raw_ids:
		if raw_id not in sum_ids:
			print(raw_id)
		if len(result_raw[raw_id].keys()) != len(result_sum[raw_id].keys()):
			print(raw_id)

	'''
	print()
	pos_raw, neg_raw = fsr(result_raw)
	print()
	pos_sum, neg_sum = fsr(result_sum)

	num_success_defense = 0
	for framing in neg_raw:
		if neg_raw not in neg_sum:
			num_success_defense = num_success_defense + 1

	print()
	print("# success defense: {}".format(num_success_defense))

def main():
	args = parse_args()
	if args.framing_response:
		framing_response(args)

if __name__ == "__main__":
	main()