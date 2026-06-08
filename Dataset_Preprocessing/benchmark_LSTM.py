from trajectory_prediction_LSTM import train_evaluate_model, prepare_datasets

list_num_epochs = [5, 10, 15]
list_batch_size = [256]

# Define the models to train based on the features they will receive
# Features index: 0=speed, 1=elapsed_time, 2=distance, 3=acceleration, 4=sin_bearing, 5=cos_bearing
models_to_train = {
    "full_model": [0, 1, 2, 3, 4, 5],
    "dist_time_bearing": [1, 2, 4, 5],
    "dist_time": [1, 2]
}

print("Preparing datasets once for all benchmarks...")
preprocessed_data = prepare_datasets()

for base_model_name, feature_indices in models_to_train.items():
    for epochs in list_num_epochs:
        for batch_size in list_batch_size:
            model_name = f"{base_model_name}_{epochs}_{batch_size}"
            model_save_path = f"Models_weights_and_results/lstm_autoencoder_{model_name}.pth"
            
            # 1. Train or load the model ONCE for this configuration (eval=False)
            print(f"\n==========================================================================")
            print(f"Loading/Training Model: {model_name}")
            print(f"==========================================================================\n")
            
            trained_model = train_evaluate_model(
                batch_size=batch_size, 
                num_epochs=epochs, 
                model_save_path=model_save_path, 
                run_eval=False,
                preprocessed_data=preprocessed_data,
                verbose=False,
                feature_indices=feature_indices,
                model_name=model_name
            )
            
            # 2. Evaluate the model once, reusing the memory-loaded model
            print(f"\n------------------------- {model_name} | Percentile: 97 --------------------------\n")
            
            train_evaluate_model(
                batch_size=batch_size, 
                num_epochs=epochs, 
                model_save_path=model_save_path, 
                anomaly_percentile=97,
                run_eval=True,
                preprocessed_data=preprocessed_data,
                pre_trained_model=trained_model,
                verbose=False,
                feature_indices=feature_indices,
                model_name=model_name
            )

# Fixed: The models and scores are now properly scoped and evaluated.
