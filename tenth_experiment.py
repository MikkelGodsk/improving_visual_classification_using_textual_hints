from image_plus_annotation_experiment import *

print("Regularization experiment (control): Weightsum based, avg. sequence. for reg=0")
print("Control: Using a 80-20 split (2 epochs). Train: WordNet,  Val: WordNet")
experiment2 = HintsExperiment(
    directory='/work3/s184399/trained_models/weightsum_unit_interval_seqavg_trainwnet_valwnet_dynamic_masking_70pct_regularization_control_experiment'
)
experiment2.regularization_rate_experiment(
    [0],
    combination_op='weighted sum',
    bert_final_op='avg sequence',
    train_kb='wordnet',
    val_kb='wordnet',
    train_val_split=0.8,
    epochs=2
)