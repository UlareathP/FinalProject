import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class FootballerPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Footballer Price Prediction")
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.Predictors = [
            'own_goal_champ', 'second_yellow_card_champ', 'red_card_champ',
            'clean_sheet_champ', 'second_yellow_card_cup', 'clean_sheet_cup',
            'own_goal_continent', 'penalty_goal_continent', 'conceded_goal_continent',
            'clean_sheet_continent', 'goal_champ', 'assist_champ', 'sub_out_champ',
            'yellow_card_champ', 'sub_on_continent'
        ]
        self.TargetVariable = 'price_log'
        self.sliders = []
        
        # Load data and model
        self.load_data_and_model()
        self.create_widgets()

    def load_data_and_model(self):
        try:
            # Load the dataset directly
            self.data = pd.read_pickle('DataForML_Numeric.pkl')
            
            # Extract features (X) and target variable (y)
            X = self.data[self.Predictors].values
            y = self.data[self.TargetVariable].values

            # Standardize the data using MinMaxScaler
            X = self.scaler.fit_transform(X)

            # Train the model
            self.model = LinearRegression()
            self.model.fit(X, y)

            messagebox.showinfo("Success", "Data loaded and model trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data or train model: {str(e)}")

    def create_widgets(self):
        # Add the predict button
        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=0, columnspan=2, pady=10)

        self.results_text = tk.Text(self.master, height=5, width=50)
        self.results_text.grid(row=1, column=0, columnspan=2, pady=20)

        for i, column in enumerate(self.Predictors):
            label = tk.Label(self.master, text=column + ': ')
            label.grid(row=i + 2, column=0)
            current_val_label = tk.Label(self.master, text='0.0')
            current_val_label.grid(row=i + 2, column=2)
            slider = ttk.Scale(self.master, from_=0, to=1, orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
            slider.grid(row=i + 2, column=1)
            self.sliders.append((slider, current_val_label))

    def predict_price(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Model is not loaded or trained properly!")
            return

        try:
            # Get input values from sliders
            inputs = np.array([[float(slider.get()) for slider, _ in self.sliders]])
            
            # Normalize the inputs as during training
            inputs = self.scaler.transform(inputs)

            # Predict using the trained model
            predicted_log_price = self.model.predict(inputs)[0]
            
            # Convert predicted log price back to original scale
            predicted_price = np.exp(predicted_log_price)
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f'The predicted player price is ${predicted_price:.2f}')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict price: {str(e)}")

if __name__ == '__main__':
    root = tk.Tk()
    app = FootballerPricePredictionApp(root)
    root.mainloop()
