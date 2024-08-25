import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class InsuranceChargesPredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Medical Insurance Charges Prediction')
        self.master.geometry('1000x950') 
        self.master.configure(bg='#f0f8ff') 

        self.data = pd.read_csv('Medical_insurance.csv')

        # Categorical columns
        self.categoricalCols = ['sex', 'smoker', 'region']

        # Features for X and targets for y
        self.x = self.data.drop('charges', axis=1)
        self.y = self.data['charges']

        # Preprocessing pipeline: One-hot encode categorical variables
        # learnt it from a youtube video and didnt know where to add the references 
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categoricalCols)],
            remainder='passthrough')

        # Create a pipeline that first preprocesses the data, then applies the model
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', RandomForestRegressor(random_state=42))])

        # Train the model
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.model.fit(self.xTrain, self.yTrain)

        self.sliders = []
        self.createWidgets()
        self.plotFeatureImportance()

        
    def createWidgets(self):
        for i, column in enumerate(self.x.columns):
            label = tk.Label(self.master, text=column + ': ', bg='#f0f8ff', font=('Arial', 12, 'bold'))
            label.grid(row=i, column=0, padx=10, pady=5, sticky='w')

            # Label for current slider value
            if column in self.categoricalCols:
                currentValLabel = tk.Label(self.master, text=self.data[column].unique()[0], bg='#f0f8ff', font=('Arial', 12))
            else:
                currentValLabel = tk.Label(self.master, text=f'{self.data[column].min():.2f}', bg='#f0f8ff', font=('Arial', 12))

            currentValLabel.grid(row=i, column=2, padx=10, pady=5, sticky='e')

            # Slider or combobox based on the type of the column
            if column in self.categoricalCols:
                slider = ttk.Combobox(self.master, values=self.data[column].unique(), font=('Arial', 12))
                slider.set(self.data[column].unique()[0])
            else:
                slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                                   command=lambda val, label=currentValLabel: label.config(text=f'{float(val):.2f}'))
                slider.configure(style='TScale')
                ttk.Style().configure('TScale', background='#e6e6e6')

            slider.grid(row=i, column=1, padx=10, pady=5)
            self.sliders.append((slider, currentValLabel))

        predictButton = tk.Button(self.master, text='Predict Charges', command=self.predictCharges, bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'))
        predictButton.grid(row=len(self.x.columns), columnspan=3, pady=10)

        featureSummaryButton = tk.Button(self.master, text='Show Feature Summary', command=self.showFeatureSummary, bg='#2196F3', fg='white', font=('Arial', 12, 'bold'))
        featureSummaryButton.grid(row=len(self.x.columns) + 1, columnspan=3, pady=5)

        resetButton = tk.Button(self.master, text='Reset Sliders', command=self.resetSliders, bg='#f44336', fg='white', font=('Arial', 12, 'bold'))
        resetButton.grid(row=len(self.x.columns) + 2, columnspan=3, pady=5)

    def showFeatureSummary(self):
        featureSummary = "\n".join([f"{name}: {value}" for name, value in zip(self.x.columns, [slider.get() for slider, _ in self.sliders])])
        messagebox.showinfo('Feature Summary', featureSummary)
            
    def resetSliders(self):
        for slider, label in self.sliders:
            if isinstance(slider, ttk.Combobox):
                # Reset combobox 
                defaultValue = slider['values'][0]
                slider.set(defaultValue)
                label.config(text=defaultValue)  
            else:
                # Reset continuous sliders
                minValue = slider.cget('from')
                slider.set(minValue)
                label.config(text=f'{minValue:.2f}') 

    def plotFeatureImportance(self):
        # Feature importance plot
        importances = self.model.named_steps['model'].feature_importances_

        # Feature names
        categoricalFeatureNames = self.model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
        nonCategoricalFeatureNames = [col for col in self.x.columns if col not in self.categoricalCols]

        featureNames = list(categoricalFeatureNames) + nonCategoricalFeatureNames
        indices = importances.argsort()[::-1]

        fig, ax = plt.subplots(figsize=(10, 6)) 
        bars = ax.bar(range(len(importances)), importances[indices], align='center', color='#4CAF50')
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([featureNames[i] for i in indices], rotation=45, ha='right') 
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().grid(row=len(self.x.columns) + 3, columnspan=3, pady=10)
        canvas.draw()

    def predictCharges(self):
        inputs = []
        for slider, _ in self.sliders:
            if isinstance(slider, ttk.Combobox):
                inputs.append(slider.get())
            else:
                inputs.append(float(slider.get()))

        inputData = pd.DataFrame([inputs], columns=self.x.columns)

        # Make predictions
        charges = self.model.predict(inputData)
        messagebox.showinfo('Predicted Charges', f'The predicted insurance charges are ${charges[0]:.2f}')


root = tk.Tk()
app = InsuranceChargesPredictionApp(root)
root.mainloop()
