from image_plus_annotation_experiment import *

print("Regularization experiment pt1: Weightsum based, avg. sequence.")
print("Using a 10-80 split (4 epochs). Train: WordNet,  Val: WordNet")
experiment2 = HintsExperiment(
    directory='/work3/s184399/trained_models/weightsum_unit_interval_seqavg_trainwnet_valwnet_dynamic_masking_70pct_regularization_experiment2'
)
experiment2.regularization_rate_experiment(
    [1e-2, 1e-1,1e+2],
    combination_op='weighted sum',
    bert_final_op='avg sequence',
    train_kb='wordnet',
    val_kb='wordnet',
    train_val_split=0.1,
    epochs=4
)