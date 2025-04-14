import numpy as np
import matplotlib.pyplot as plt
from floris import FlorisModel

def prediction(ti_recorded_new, wind_speed_recorded_new, wind_direction_recorded_new, fmodel_true, power_recorded, U_recorded, layout_x, layout_y, wind_shear,phi_recorded):

    # Stima iniziale (ipotesi di partenza, tipicamente da meteo o dati storici)
    wind_direction_test = wind_direction_recorded_new   # direzione iniziale stimata
    ws_test = wind_speed_recorded_new                  # velocità iniziale stimata
    ti_test = ti_recorded_new              # turbolenza iniziale stimata

    print('Condizioni iniziali stimate:')
    print(f'Stima iniziale direzione: {wind_direction_test:.2f}°')
    print(f'Stima iniziale velocità: {ws_test:.2f} m/s')
    print(f'Stima iniziale turbolenza: {ti_test:.4f}')

    # Range di esplorazione intorno alla stima iniziale (spazio di ricerca = delta)
    delta_phi_range = np.linspace(-20, 20, 21)
    delta_ws_range = np.linspace(-1.5, 1.5, 11)
    delta_ti_range = np.linspace(0, 0, 1)
    phi_range = wind_direction_test + delta_phi_range
    ws_range = ws_test + delta_ws_range
    ti_range = ti_test + delta_ti_range

    J_array = np.zeros((len(delta_phi_range), len(delta_ws_range), len(delta_ti_range)))

    lambda_P = 1
    lambda_U = 0
    lambda_phi = 0 # Non serve: non stai confrontando direzione con direzione reale.

    fmodel_est = FlorisModel(r"..\\examples\\inputs\\cc_ATIS_polito.yaml")
    fmodel_est.set(
        layout_x=layout_x,
        layout_y=layout_y,
        wind_shear=wind_shear,
    )

    for i, delta_phi in enumerate(delta_phi_range):
        for j, delta_ws in enumerate(delta_ws_range):
            for k, delta_ti in enumerate(delta_ti_range):
                phi_hat = wind_direction_test + delta_phi
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

                err_p = np.sum((power_recorded - power_hat)**2)
                err_u = np.sum((U_recorded - U_hat)**2)
                err_phi = np.sum((phi_recorded - wd_hat)**2)

                J = (1 / len(layout_x)) * (lambda_P * err_p + lambda_U * err_u + lambda_phi * err_phi)
                J_array[i, j, k] = J

    # Find the indices of the minimum cost function
    prediction_index = np.argmin(J_array)
    delta_phi_index, delta_ws_index, delta_ti_index = np.unravel_index(prediction_index, J_array.shape)
    delta_phi = delta_phi_range[delta_phi_index]
    delta_ws = delta_ws_range[delta_ws_index]
    delta_ti = delta_ti_range[delta_ti_index]

    wind_direction_test = wind_direction_test + delta_phi
    ws_test = ws_test + delta_ws
    ti_test = ti_test + delta_ti

    print('Condizioni stimate finali:')
    print(f'Stima finale direzione: {wind_direction_test:.2f}°')
    print(f'Stima finale velocità: {ws_test:.2f} m/s')
    print(f'Stima finale turbolenza: {ti_test:.4f}')

    # Plot the cost function with respect to delta_phi
    J_slice = J_array[:, delta_ws_index, delta_ti_index]  # Fix delta_ws and delta_ti
    plt.figure(figsize=(10, 6))
    plt.plot(delta_phi_range, J_slice, label="Cost Function J", color="blue", lw=2)
    plt.axvline(delta_phi, color="red", linestyle="--", label=f"Min Delta Phi = {delta_phi:.2f}")
    plt.axhline(np.min(J_slice), color="green", linestyle="--", label=f"Min J = {np.min(J_slice):.4f}")
    plt.xlabel("Delta Phi (degrees)")
    plt.ylabel("Cost Function J")
    plt.title("Cost Function J vs Delta Phi")
    plt.legend()
    plt.grid(True)
    plt.show()

    return wind_direction_test, ws_test, ti_test, wind_direction_test, ws_test, ti_test
