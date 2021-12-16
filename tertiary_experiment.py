from image_plus_annotation_experiment import *

if __name__ == '__main__':
    """experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/concatenation_avgseq_trainwnet_valwnet_dynamic_masking_70pct'
    )

    print("Concatenation + avg. seq. Regularization rate = 0.")
    print("Training on WordNet (p_kept=0.7), validating on WordNet. Testing on both individually")
    experiment1.new_model(
        regularization_rate=0.0,
        combination_op='concatenation',
        bert_final_op='avg sequence'
    )
    ds_train = DatasetHandler(
        train_kb='wordnet',
        val_kb='wordnet',
        test_kb='wordnet'
    )
    ds_val = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wikipedia',
        test_kb='wikipedia'
    )
    ds_val.drop_observations_without_hints(for_all_partitions=True)
    experiment1.dataset = ds_train
    experiment1.train_model(epochs=2, dynamic_masking=True, dynamic_p=tf.constant(0.7))
    print("Evaluate on test set: WordNet")
    experiment1.evaluate(evaluate_on='test')
    experiment1.dataset = ds_val
    print("Evaluate on test set: Wikipedia")
    experiment1.evaluate(evaluate_on='test')
    experiment1.save_model()"""

    print("Regularization experiment (control): Weightsum based, avg. sequence. for reg=0")
    print("Control: Using a 80-20 split")
    experiment2 = HintsExperiment(
        directory='/work3/s184399/trained_models/weightsum_seqavg_trainwiki_valwnet_dynamic_masking_70pct_control'
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

