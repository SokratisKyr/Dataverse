import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import shap
import tkinter as tk
from tkinter import ttk
import numpy as np


def load_data(filepath):
    return pd.read_csv(filepath)


def calculate_statistics(df):
    selected_columns = ['value', 'value_type_id', 'location_id', 'source_id']
    stats = df[selected_columns].agg(['min', 'mean', 'max'])
    print("\nMax, min values, and average for columns `value`, `value_type_id`, `location_id`, `source_id`:\n")
    print(stats)


def categorical_frequencies(df):
    print("\nFrequency of Categorical Columns:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n{col}:")
        if col == 'timestamp':
            df[col] = pd.to_datetime(df[col])
            frequency_per_minute = df[col].dt.to_period('T').value_counts().sort_index()
            print("Frequency per minute:")
            print(frequency_per_minute)
            continue
        print(df[col].value_counts())


def encode_categorical_data(df):
    # One-hot encode the categorical feature 'identifier'
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded_data = onehot_encoder.fit_transform(df[['identifier']])
    onehot_df = pd.DataFrame(onehot_encoded_data, columns=onehot_encoder.get_feature_names_out(['identifier']))
    df = pd.concat([df.reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)
    return df, onehot_encoder


def process_timestamps(df):
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second


def scale_numerical_data(df):
    scaler_features = StandardScaler()
    scaler_value = StandardScaler()

    # Scale features (excluding 'value')
    columns_to_scale = ['location_id', 'source_id', 'year', 'month', 'day', 'hour']
    df[columns_to_scale] = scaler_features.fit_transform(df[columns_to_scale])

    # Scale 'value' separately
    df[['value']] = scaler_value.fit_transform(df[['value']])

    return df, scaler_features, scaler_value


def create_model(num_layers, neurons_per_layer, activation_functions, error_function, optimization_method):
    model = keras.Sequential()
    # Add each layer based on user's input
    for i in range(num_layers):
        model.add(layers.Dense(neurons_per_layer[i], activation=activation_functions[i]))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimization_method, loss=error_function)
    return model


def show_results_window(results_df):
    window = tk.Tk()
    window.title("Model Evaluation Results")

    tree = ttk.Treeview(window)
    tree.pack(side='top', fill='both', expand=True)

    tree["columns"] = list(results_df.columns)
    tree["show"] = "headings"

    for col in results_df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    for index, row in results_df.iterrows():
        tree.insert("", "end", values=list(row))

    window.geometry("800x400")
    window.mainloop()


def make_prediction(model, scaler_features, scaler_value, onehot_encoder, df):
    print("\nEnter values for prediction:")

    # Define valid values from the dataset for location_id, source_id, and identifier
    valid_location_ids = df['location_id'].unique()
    valid_source_ids = df['source_id'].unique()
    valid_identifiers = onehot_encoder.categories_[0]

    # Display available options and get inputs
    print(f"Available location IDs: {valid_location_ids}")
    while True:
        location_id = float(input("Enter location_id: "))
        if location_id in valid_location_ids:
            break
        print("Invalid location_id. Please choose from the list above.")

    print(f"\nAvailable source IDs: {valid_source_ids}")
    while True:
        source_id = float(input("Enter source_id: "))
        if source_id in valid_source_ids:
            break
        print("Invalid source_id. Please choose from the list above.")

    print(f"\nAvailable identifiers: {valid_identifiers}")
    while True:
        identifier = input("Enter identifier: ")
        if identifier in valid_identifiers:
            break
        print("Invalid identifier. Please choose from the list above.")

    while True:
        year = int(input("Enter year: "))
        if 2009 <= year <= 2009:
            break
        print("Year must be 2009.")

    while True:
        month = int(input("Enter month (1-12): "))
        if 1 <= month <= 12:
            break
        print("Month should be between 1 and 12.")

    while True:
        day = int(input("Enter day (1-31): "))
        if 1 <= day <= 31:
            break
        print("Day should be between 1 and 31.")

    while True:
        hour = int(input("Enter hour (0-23): "))
        if 0 <= hour <= 23:
            break
        print("Hour should be between 0 and 23.")

    # One-hot encode the identifier
    onehot_encoded = onehot_encoder.transform([[identifier]])
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['identifier']))

    # Combine inputs into a DataFrame
    input_df = pd.DataFrame({
        'location_id': [location_id],
        'source_id': [source_id],
        'year': [year],
        'month': [month],
        'day': [day],
        'hour': [hour]
    })

    # Add one-hot encoded columns
    input_df = pd.concat([input_df, onehot_encoded_df], axis=1)

    # Scale the features that the scaler expects
    input_df_scaled = input_df.copy()
    input_df_scaled[['location_id', 'source_id', 'year', 'month', 'day', 'hour']] = scaler_features.transform(
        input_df[['location_id', 'source_id', 'year', 'month', 'day', 'hour']]
    )

    # Make prediction using the scaled input DataFrame
    prediction = model.predict(input_df_scaled)
    predicted_value = prediction[0][0]
    predicted_value_original_scale = scaler_value.inverse_transform([[predicted_value]])[0][0]
    print(f"\nPredicted value (original scale): {predicted_value_original_scale}")
    print(f"\nPredicted value: {prediction[0][0]}")


def main():
    df = load_data('C:/Users/sokky/Downloads/ML-assignment/wheather_data.csv')
    initial_df = df

    calculate_statistics(df)
    categorical_frequencies(df)
    df, onehot_encoder = encode_categorical_data(df)
    process_timestamps(df)
    df, scaler_features, scaler_value = scale_numerical_data(df)

    # Split dataset to input->X, output->Y
    onehot_columns = [col for col in df.columns if col.startswith("identifier_")]
    X = df[['location_id', 'source_id', 'year', 'month', 'day', 'hour'] + onehot_columns]
    Y = df['value'].values

    results = []

    while True:
        while True:
            try:
                num_layers = int(input("\n\nEnter the number of the desired network layers: "))
                break
            except ValueError:
                print("Please enter a valid number.")

        neurons_per_layer = []
        activation_functions = []
        for i in range(num_layers):
            while True:
                try:
                    neurons = int(input(f"Enter the number of neurons for layer {i + 1}: "))
                    neurons_per_layer.append(neurons)
                    break
                except ValueError:
                    print("Please enter a valid number.")

            while True:
                activation = input(f"Enter activation function for layer {i + 1} (sigmoid/relu): ")
                if activation.lower() in ['sigmoid', 'relu']:
                    activation_functions.append(activation)
                    break
                else:
                    print("Please choose 'sigmoid' or 'relu'.")

        while True:
            error_function = input("Choose the error function (1: mean_squared_error / 2: mean_absolute_error): ")
            if error_function == '1':
                error_function = 'mean_squared_error'
                break
            elif error_function == '2':
                error_function = 'mean_absolute_error'
                break
            else:
                print("Please choose '1' for mean_squared_error or '2' for mean_absolute_error.")

        while True:
            optimization_method = input("Choose training method (sgd/adam): ")
            if optimization_method.lower() in ['sgd', 'adam']:
                break
            else:
                print("Please choose 'sgd' or 'adam'.")

        # Initialize model
        model = create_model(num_layers, neurons_per_layer, activation_functions, error_function, optimization_method)

        # Stop training when validation loss doesn't improve
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Split dataset to train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Train model with early stopping
        model.fit(X_train, Y_train,
                  validation_data=(X_test, Y_test),
                  epochs=100,
                  batch_size=32,
                  callbacks=[early_stopping])

        loss = model.evaluate(X_test, Y_test)
        print(f"Model loss: {loss}")

        # Save results of each network to be printed in the end
        results.append({
            "num_layers": num_layers,
            "neurons_per_layer": neurons_per_layer,
            "activation_functions": activation_functions,
            "error_function": error_function,
            "optimization_method": optimization_method,
            "loss": loss
        })

        # SHAP analysis
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_train, 10))
        shap_values = explainer.shap_values(X_test[:100])

        shap_values_array = np.array(shap_values[0]).flatten()
        feature_names = X_test.columns.tolist()
        shap.bar_plot(shap_values_array, feature_names=feature_names)

        # Ask if the user wants to make a prediction
        prediction_bool = input("Do you want to make a prediction? (yes/no): ")
        if prediction_bool.lower() == 'yes':
            while True:
                make_prediction(model, scaler_features, scaler_value, onehot_encoder, initial_df)

                another_prediction = input("Do you want to make another prediction? (yes/no): ")
                if another_prediction.lower() != 'yes':
                    break
        else:
            print("No predictions will be made.\n")

        continue_training = input("\nDo you want to create a new model? (yes/no): ")
        if continue_training.lower() != 'yes':
            break

    results_df = pd.DataFrame(results)
    print("\nModel Evaluation Results:")
    print(results_df)

    show_results_window(results_df)


if __name__ == "__main__":
    main()
