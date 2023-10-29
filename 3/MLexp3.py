
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('placement.csv')

# Data visualization
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')

# Prepare data for training
X = df.iloc[:, 0:1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# GUI setup
root = tk.Tk()
root.title('Placement Package Predictor')
root.geometry('800x400')

# Plotting
fig = plt.Figure(figsize=(5, 5))
plt.scatter(df['cgpa'], df['package'])
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# User interface
frame_right = tk.Frame(root)
frame_right.pack(side=tk.LEFT, padx=20)

label_cgpa = tk.Label(frame_right, text='Enter CGPA:')
label_cgpa.pack()

entry_cgpa = tk.Entry(frame_right)
entry_cgpa.pack()

def predict_package():
    try:
        cgpa = float(entry_cgpa.get())
        predicted_package = lr.predict(np.array([[cgpa]]))[0]
        label_predicted.config(text=f'Predicted Package: {predicted_package:.2f} LPA')
    except ValueError:
        label_predicted.config(text="Invalid input. Please enter a valid CGPA.")

btn_predict = tk.Button(frame_right, text='Predict', command=predict_package)
btn_predict.pack()

label_predicted = tk.Label(frame_right, text='Predicted Package:')
label_predicted.pack()

root.mainloop()
