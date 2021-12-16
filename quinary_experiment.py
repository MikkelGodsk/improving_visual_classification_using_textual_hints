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
        val_kb='wordnet',
        test_kb='wordnet',
        train_val_split=0.2
    )
    ds_val.drop_observations_without_hints(for_all_partitions=True)

    experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/less_data/addition_avgseq_trainwiki_valwiki_testwikiwnet_dynamic_masking_70pct'
    )
    print("Addition + avg. seq. Regularization rate = 0.0")
    print("Training on Wikipedia (p_kept=0.7), validating on Wikipedia. Testing on both individually")
    print("Dataset: 0.2 train - 0.8 validation")
    experiment1.new_model(
        combination_op='addition',
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
    experiment1.save_model()

    ds_wn = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wordnet',
        test_kb='wikipedia',
        train_val_split=0.8
    ).drop_observations_without_hints(for_all_partitions=True)
    ds_wiki = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wikipedia',
        test_kb='wordnet',
        train_val_split=0.8
    ).drop_observations_without_hints(for_all_partitions=True)
    print("Conducting the mask proportion experiment (WordNet)")
    experiment1.dataset = ds_wn
    experiment1.experiment_results = []
    print(experiment1.mask_proportion_experiment())
    print("Conducting the mask proportion experiment (Wikipedia)")
    experiment1.dataset = ds_wiki
    experiment1.experiment_results = []
    print(experiment1.mask_proportion_experiment())"""
    """
    print("Training and evaluating Weightsum based (l2 rate = 1e-2), avg. sequence.")
    experiment2 = HintsExperiment(
        directory='/work3/s184399/trained_models/weightsum_seqavg_trainwiki_valwnet_dynamic_masking_70pct_reg_1e-2'
    )
    experiment2.get_dataset(train_kb='wikipedia', val_kb='wordnet', train_val_split=0.8)
    experiment2.dataset.drop_observations_without_hints(for_all_partitions=True)
    experiment2.new_model(regularization_rate=1e-2, bert_final_op='avg sequence', combination_op='weighted sum')
    experiment2.train_model(epochs=2, dynamic_masking=True, dynamic_p=tf.constant(0.7))
    """

    print("Learning curve experiment: Weighted sum (reg: 1e-5) + avg. seq. Weights in unit interval")
    experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/less_data/weighted_sum_avgseq_learningcurve_dynamic_masking_70pct'
    )
    experiment1.learning_curve_experiment(bert_final_op='avg sequence', combination_op='weighted sum', regularization_rate=1e-4)