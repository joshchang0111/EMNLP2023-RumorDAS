#for dataset in semeval2019 twitter15 twitter16
#do
#	python others/select_examples.py --select_for_user_study --dataset_name $dataset
#done

#python others/select_examples.py --generate_for_google_forms
#python others/select_examples.py --merge_en_zh
#python others/select_examples.py --distribute_samples_for_diff_tasks
#python others/select_examples.py --select_which_to_swap

#python others/human_evaluation.py --evaluate_A1
#python others/human_evaluation.py --evaluate_A2

#python others/human_evaluation.py --evaluate_B1
#python others/human_evaluation.py --evaluate_B2

#python others/human_evaluation.py --box_plot_A
#python others/human_evaluation.py --box_plot_AB

#python others/human_evaluation.py --bar_plot_B

python others/human_evaluation.py --agreement_analysis