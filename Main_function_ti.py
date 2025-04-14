import numpy as np
import matplotlib.pyplot as plt
from floris import FlorisModel
from tqdm import tqdm  # Import tqdm for progress bars

def prediction(ti_recorded_new, wind_speed_recorded_new, wind_direction_recorded_new, fmodel_true, power_recorded, U_recorded, layout_x, layout_y, wind_shear,phi_recorded):

    # Stima iniziale (ipotesi di partenza, tipicamente da meteo o dati storici)
    wind_direction_test = wind_direction_recorded_new   # direzione iniziale stimata
    ws_test = wind_speed_recorded_new                  # velocità iniziale stimata
    ti = ti_recorded_new              # turbolenza iniziale stimata

    print('Condizioni iniziali stimate:')
    print(f'Stima iniziale direzione: {wind_direction_test:.2f}°')
    print(f'Stima iniziale velocità: {ws_test:.2f} m/s')
    print(f'Stima iniziale turbolenza: {ti:.4f}')

    # Range di esplorazione intorno alla stima iniziale (spazio di ricerca = delta)
    true_wind = np.linspace(0, 360, 61)
    delta_phi_range = np.linspace(-10, 10, 11)
    delta_ws_range = np.linspace(-1.5, 1.5, 11)
    delta_ti_range = np.linspace(0, 0, 1)
    ti_values = [0.065, 0.095, 0.125, 0.155]  


    print(len(true_wind))
    phi_range = wind_direction_test + delta_phi_range
    ws_range = ws_test + delta_ws_range
    ti_range = ti + delta_ti_range

    J_array = np.zeros((len(delta_phi_range), len(delta_ws_range), len(delta_ti_range)))
    M_array = np.zeros((len(true_wind), len(delta_phi_range), len(delta_ws_range), len(delta_ti_range)))

    lambda_P = 1e-12
    lambda_U = 1
    lambda_phi = 10  # Non serve: non stai confrontando direzione con direzione reale.

    fmodel_est = FlorisModel(r"..\\examples\\inputs\\cc_ATIS_polito.yaml")
    fmodel_est.set(
        layout_x=layout_x,
        layout_y=layout_y,
        wind_shear=wind_shear,
    )

    b_phi = 1.0
    b_ws = 0.1
    b_I = 0.01
    O_val = np.zeros((len(ti_values),len(true_wind)))  # Adjust for multiple ti values
    k_phi = 1.0
    k_U = 1.0
    k_I = 1.0

    for ti_index, ti_test in enumerate(ti_values):  # Outer loop for ti values
        print(f"Processing turbulence intensity: {ti_test:.4f}")
        # Use tqdm to create a progress bar for the wind loop
        for t, wind in enumerate(tqdm(true_wind, desc=f"TI={ti_test:.4f} Progress", unit="wind")):
            fmodel_est.set(
                wind_directions=[wind],
                wind_speeds=[ws_test],
                turbulence_intensities=[ti_test]
            )
            fmodel_est.run()
            power_true = fmodel_est.get_turbine_powers().flatten() / 200
            U_true = fmodel_est.turbine_average_velocities.flatten()
            wd_true = fmodel_est.wind_directions 
            for i, delta_phi in enumerate(delta_phi_range):
                for j, delta_ws in enumerate(delta_ws_range):
                    for k, delta_ti in enumerate(delta_ti_range):
                    
                        phi_hat = wind + delta_phi
                        ws_hat = ws_test + delta_ws
                        ti_hat = ti_test + delta_ti

                        fmodel_est.set(
                            wind_directions=[phi_hat],
                            wind_speeds=[ws_hat],
                            turbulence_intensities=[ti_hat]
                        )
                        fmodel_est.run()

                        power_hat = fmodel_est.get_turbine_powers().flatten() / 200
                        U_hat = fmodel_est.turbine_average_velocities.flatten()
                        wd_hat = fmodel_est.wind_directions
                        
                        err_p = np.sum((power_true - power_hat)**2)
                        err_u = np.sum((U_true - U_hat)**2)
                        err_phi = np.sum((wd_true - wd_hat)**2)

                        J = (1 / len(layout_x)) * (lambda_P * err_p + lambda_U * err_u + lambda_phi * err_phi)
                        J_array[i, j, k] = J

                        # Calcolo osservabilità
                        d_phi = abs(delta_phi)
                        d_u = abs(delta_ws)
                        d_i = abs(delta_ti)

                        if (d_phi < b_phi) and (d_u < b_ws) and (d_i < b_I):
                            M_array[t, i, j, k] = np.inf
                        else:
                            denom = k_phi * (d_phi**2) + k_U * (d_u**2) + k_I * (d_i**2)
                            M_array[t, i, j, k] = J / max(denom, 1e-6)
            # Fisso velocità e turbolenza
            O_val[ti_index,t] = np.min(M_array[t])  # Store the minimum observability value for each wind direction
    
    O_val_normalized = O_val / np.max(O_val[ti_index])
    print(O_val_normalized.shape)
    print('Osservabilità minima:', np.min(O_val_normalized))
    
    # Create a single figure with subplots for all turbulence intensities
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 10))  # 2x2 grid of polar plots
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for ti_index, ti_test in enumerate(ti_values):
        wind_directions_rad = np.deg2rad(true_wind)  # Normalize M_slice for plotting
        ax = axes[ti_index]
        c = ax.scatter(wind_directions_rad, O_val_normalized[ti_index], c=O_val_normalized[ti_index], cmap="viridis", edgecolors="k")
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)  # Clockwise direction
        ax.set_title(f"TI={ti_test:.4f}", va='bottom')
        fig.colorbar(c, ax=ax, label="Observability")

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

    return J_array, M_array, O_val