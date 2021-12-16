from image_plus_annotation_experiment import *

if __name__ == '__main__':
    """
    ds_train = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wikipedia',
        test_kb='wikipedia',
        train_val_split=0.2
    )
    ds_val = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wikipedia',
        test_kb='wordnet',
        train_val_split=0.2
    )
    ds_val.drop_observations_without_hints(for_all_partitions=True)

    experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/less_data/concatenation_avgseq_trainwiki_valwiki_testwikiwnet_dynamic_masking_70pct'
    )
    print("Concatenation + avg. seq. Regularization rate = 0.")
    print("Training on Wikipedia (p_kept=0.7), validating on Wikipedia. Testing on both individually")
    print("Dataset: 0.2 train - 0.8 validation")
    experiment1.new_model(
        regularization_rate=0.0,
        combination_op='concatenation',
        bert_final_op='avg sequence'
    )
    experiment1.dataset = ds_train
    experiment1.train_model(epochs=1, dynamic_masking=True, dynamic_p=tf.constant(0.7))
    print("Evaluate on test set: Wikipedia")
    experiment1.evaluate(evaluate_on='test')
    experiment1.dataset = ds_val
    print("Evaluate on test set: WordNet")
    experiment1.evaluate(evaluate_on='test')
    experiment1.save_model()"""

    print("Training and evaluating Weightsum based (no regularization), avg. sequence.")
    experiment2 = HintsExperiment(
        directory='/work3/s184399/trained_models/weightsum_seqavg_trainwiki_valwnet_dynamic_masking_70pct'
    )
    experiment2.get_dataset(train_kb='wikipedia', val_kb='wordnet', train_val_split=0.8)
    experiment2.dataset.drop_observations_without_hints(for_all_partitions=True)
    experiment2.new_model(regularization_rate=0.0, bert_final_op='avg sequence', combination_op='weighted sum')
    experiment2.train_model(epochs=2, dynamic_masking=True, dynamic_p=tf.constant(0.7))