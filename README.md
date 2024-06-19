# PINN Project

This project implements a Physics-Informed Neural Network (PINN) to solve the Burgers' equation. The project is structured around two main files: `nn.py` and `solve.py`.

## Project Structure

### Files

- **nn.py**: Contains the main code to define constants, generate training points, and a function to solve the Burgers' equation.
- **solve.py**: Contains similar functions to generate training points and solve the Burgers' equation, but is used as a separate solving script.

### Explanation of Functions and Variables in solve.py

Constants:
    pi, viscosity, tmin, tmax, xmin, xmax, lb, ub: Constants used to define the domain boundaries and parameters of the Burgers' equation.
Condition Functions:
    fun_u_0(x): Function defining the initial condition.
    fun_u_b(t, x): Function defining the boundary conditions.
Point Generation:
    t_0, x_0, X_0, u_0: Points and values for the initial condition.
    t_b, x_b, X_b, u_b: Points and values for the boundary conditions.
    t_r, x_r, X_r, u_r: Points in the equation's domain.
    X_data, u_data: Combined data for the initial and boundary conditions.
Visualization Functions:
    plot_train_points: Function to visualize the points used during the training phase of the PINN.
Solving Function:
    solve_burger: Function to set up and train the PINN model, then solve the Burgers' equation. It saves the model weights, generates visualizations, and prints performance metrics.

## Usage

### Dependencies

### Running the Code

To solve the Burgers' equation using the PINN, use the following command in your terminal:
```bash
python solve.py
```

### Note

The results are highly dependent on the hyperparameters specified in the `solve_burger` function. The default execution does not guarantee satisfactory results.

## Results

The training results, including solution and loss graphs, will be saved in the plot/ directory. The model weights will be saved in the weights/ directory.

## Contributors

    Main Author: cycymonkey

## Licence 

