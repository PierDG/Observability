import numpy as np
import matplotlib.pyplot as plt
from floris import FlorisModel

def prediction(ti_recorded_new, wind_speed_recorded_new, wind_direction_recorded_new, fmodel_true, power_recorded, U_recorded, layout_x, layout_y, wind_shear, aaaa):

    # Stima iniziale (ipotesi di partenza, tipicamente da meteo o dati storici)
    wind_direction_test = wind_direction_recorded_new   # direzione iniziale stimata
    ws_test = wind_speed_recorded_new                  # velocità iniziale stimata
    ti_test = ti_recorded_new              # turbolenza iniziale stimata

    print('Condizioni iniziali stimate:')
    print(f'Stima iniziale direzione: {wind_direction_test:.2f}°')
    print(f'Stima iniziale velocità: {ws_test:.2f} m/s')
    print(f'Stima iniziale turbolenza: {ti_test:.4f}')

    # Range di esplorazione intorno alla stima iniziale (spazio di ricerca = delta)
    true_wind = np.linspace(0, 360, 61)
    delta_phi_range = np.linspace(-20, 20, 21)
    delta_ws_range = np.linspace(-1.5, 1.5, 11)
    delta_ti_range = np.linspace(0, 0, 1)
    print(len(true_wind))
    phi_range = wind_direction_test + delta_phi_range
    ws_range = ws_test + delta_ws_range
    ti_range = ti_test + delta_ti_range

    J_array = np.zeros((len(delta_phi_range), len(delta_ws_range), len(delta_ti_range)))
    M_array = np.zeros((len(true_wind), len(delta_phi_range), len(delta_ws_range), len(delta_ti_range)))

    lambda_P = 1e-12
    lambda_U = 1
    lambda_phi = 10   # Non serve: non stai confrontando direzione con direzione reale.

    fmodel_est = FlorisModel(r"..\\examples\\inputs\\cc_ATIS_polito.yaml")
    fmodel_est.set(
        layout_x=layout_x,
        layout_y=layout_y,
        wind_shear=wind_shear,
    )

    b_phi = 1.0
    b_ws = 0.1
    b_I = 0.01
    O_val = np.zeros(len(true_wind))
    k_phi = 1.0
    k_U = 1.0
    k_I = 1.0

    for t, wind in enumerate(true_wind):
        fmodel_est.set(
            wind_directions=[wind],
            wind_speeds=[ws_test],
            turbulence_intensities=[ti_test]
        )
        fmodel_est.run()
        power_true = fmodel_est.get_turbine_powers().flatten()/200
        U_true = fmodel_est.turbine_average_velocities.flatten()
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

                    power_hat = fmodel_est.get_turbine_powers().flatten()/200
                    U_hat = fmodel_est.turbine_average_velocities.flatten()
                    wd_hat = fmodel_est.wind_directions
                    
                    err_p = np.sum((power_true - power_hat)**2)
                    err_u = np.sum((U_true - U_hat)**2)
                    err_phi = np.sum((wind - wd_hat)**2)

                    J = (1 / len(layout_x)) * (lambda_P * err_p + lambda_U * err_u+lambda_phi * err_phi)
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
        # J_slice = J_array[:, 0, 0]
        # M_slice = M_array[t, :, 0, 0]
        # maxind = np.argmax(M_array)
        # if M_array[maxind] == np.inf:
        #     M_array[maxind] = np.min(M_slice)
        # M_slice_normalized = M_slice / np.max(M_slice)
        O_val[t] = np.min(M_array[t])
        print(f"Completamento:{(t+1)/61*100:.6f}%")
    O_val_normalized = O_val / np.max(O_val)
    min_index = np.unravel_index(np.argmin(M_array), M_array.shape)
    print(min_index)
    print(O_val_normalized, len(O_val_normalized))
    print('Osservabilità minima:', np.min(O_val))
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ax_left = axes[0]
    # ax_left.plot(delta_phi_range, J_slice, 'k', lw=1.5)
    # ax_left.axvline(0, color='r', linestyle='--', label='Real Δphi = 0')
    # ax_left.axvline(delta_phi_range[min_index[0]], color='b', linestyle='--', label=f'Critical Δphi = {delta_phi_range[min_index[0]]:.2f}°')
    # ax_left.set_xlabel(r'$\Delta \phi$ (deg)')
    # ax_left.set_ylabel(r'$J(\Delta \phi)$')
    # ax_left.set_title('Cost function J')
    # ax_left.legend()
    # ax_left.grid(True)

    # ax_right = axes[1]
    # ax_right.plot(delta_phi_range, M_slice, 'k', lw=1.5)
    # ax_right.axvline(0, color='r', linestyle='--', label='Real Δphi = 0')
    # ax_right.axvline(delta_phi_range[min_index[0]], color='b', linestyle='--', label=f'Critical Δphi = {delta_phi_range[min_index[0]]:.2f}°')
    # ax_right.set_xlabel(r'$\Delta \phi$ (deg)')
    # ax_right.set_ylabel(r'$\mathcal{M}(J)$')
    # ax_right.set_title('Observability M')
    # ax_right.legend()
    # ax_right.grid(True)

    # plt.tight_layout()
    # plt.show()
 # Convert degrees to radians for the polar plot
    wind_directions_rad = np.deg2rad(true_wind)  # Normalize M_slice for plotting
    print(len(wind_directions_rad), len(O_val_normalized))
    # Create the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    c = ax.scatter(wind_directions_rad, O_val_normalized, c=O_val_normalized, cmap="viridis", edgecolors="k")
    # Labels and formatting
    ax.set_theta_zero_location('N')  # North at top
    ax.set_theta_direction(-1)  # Clockwise direction
    ax.set_title("Wind Direction Observability", va='bottom')

    # Add colorbar
    fig.colorbar(c, ax=ax, label="Observability")

    plt.show()
    return J_array, M_array, O_val