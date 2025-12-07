# Linear Regression: The Math Behind the Model (Interview Notes)

## 1. The Core Equation
Linear Regression tries to fit a straight line through the data to predict a continuous value (like Insurance Cost).
**Formula:** $y = wx + b$

* **$y$ (Prediction):** The target variable (Cost).
* **$x$ (Input):** The features (Age, BMI, Smoking).
* **$w$ (Weight):** How important the input is. (e.g., If Smoking has a high weight, it increases the cost heavily).
* **$b$ (Bias):** The baseline cost (the price even if all inputs were zero).

## 2. How the Model "Learns" (Cost Function)
The computer guesses a random line, then measures how wrong it is using **MSE (Mean Squared Error)**.
* **Error:** The difference between the *Predicted Cost* and the *Actual Cost*.
* **Goal:** Minimize the MSE. Find the $w$ and $b$ that make the total error as small as possible.

## 3. The Optimization: Gradient Descent
Imagine standing on a mountain blindfolded (High Error). You want to walk down to the valley (Lowest Error).
* **Gradient:** The slope of the hill.
* **Descent:** The model calculates the slope mathematically and takes a "step" downhill.
* It repeats this thousands of times until it reaches the bottom (the best possible line).

## 4. Why this matters for Insurance?
We want to find the exact "weight" of smoking. If $w_{smoking} = 20000$, it means being a smoker adds $20,000 to your bill mathematically.