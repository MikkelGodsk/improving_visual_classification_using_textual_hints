from image_plus_annotation_experiment import *

if __name__ == '__main__':
    """
    ds_train = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wikipedia',
        test_kb='wikipedia',
        train_val_split=0.8
    )
    ds_val = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wordnet',
        test_kb='wordnet',
        train_val_split=0.8
    )
    ds_val.drop_observations_without_hints(for_all_partitions=True)

    experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/less_data/weighted_sum_reg_avgseq_trainwiki_valwiki_testwikiwnet_dynamic_masking_70pct'
    )
    print("Weighted sum + avg. seq. Regularization rate = 1e-2")
    print("Training on Wikipedia (p_kept=0.7), validating on Wikipedia. Testing on both individually")
    print("Dataset: 0.8 train - 0.2 validation")
    experiment1.new_model(
        regularization_rate=1e-2,
        combination_op='weighted sum',
        bert_final_op='avg sequence'
    )
    experiment1.dataset = ds_train
    experiment1.train_model(epochs=1, dynamic_masking=True, dynamic_p=tf.constant(0.7))
    print("Evaluate on test set: Wikipedia")
    experiment1.evaluate(evaluate_on='test')
    experiment1.dataset = ds_val
    print("Evaluate on WordNet validation")
    experiment1.evaluate(evaluate_on='validation')
    print("Evaluate on test set: WordNet")
    experiment1.evaluate(evaluate_on='test')
    experiment1.save_model()"""

    print("Learning curve experiment: Weighted sum + avg. seq. Weights in unit interval")
    experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/less_data/weighted_sum_avgseq_learningcurve_dynamic_masking_70pct'
    )
    experiment1.learning_curve_experiment(bert_final_op='avg sequence', combination_op='weighted sum')

