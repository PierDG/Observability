
import matplotlib.pyplot as plt

from floris import (
    FlorisModel,
)
import sys
sys.path.append(r'C:\Users\pierp\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\FLORIS-main\FLORIS-main\floris-main_1\closed_loop\__pycache__')
import Main_function_ti as fun
import floris.layout_visualization as layoutviz
from floris.flow_visualization import visualize_cut_plane

fmodel_true = FlorisModel(r"..\\examples\inputs\cc_ATIS_polito.yaml")
# Layout Settings
tworows = False
onerow = True
rnd = False

wind_speed_recorded = [9.00]
wind_direction_recorded = [10.0]   
ti_recorded = [0.055]
D = 240/200
wind_direction_recorded_new = 10.0 
wind_speed_recorded_new = 9.00
if tworows == True:
  layout_x = [0., 5*D, 10*D, 0., 5*D, 10*D]
  layout_y = [0., 0., 0., 3.5*D, 3.5*D, 3.5*D]
elif onerow == True:
    layout_x = [0., 5*D, 10*D]
    layout_y = [0., 0., 0.]
elif rnd == True:
      layout_x = [0., -2*D , -4.5*D,-7.5*D,-6*D, 1*D] 
      layout_y = [0., -4*D, -1.5*D,-3.75*D,-9*D, -8*D]

wind_shear = 0.06

fmodel_true.set(
    layout_x=layout_x,
    layout_y=layout_y,
    wind_speeds = wind_speed_recorded,
    wind_directions = wind_direction_recorded,
    turbulence_intensities = ti_recorded,
    turbine_library_path= r'C:\Users\pierp\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\FLORIS-main\FLORIS-main\floris-main_1\floris\turbine_library',
    turbine_type= ['iea_15Mw_bot_fixed_lqr'],
    wind_shear=0.06,
)

fmodel_true.run()

power_recorded = (fmodel_true.get_turbine_powers()).flatten()/200
U_recorded = (fmodel_true.turbine_average_velocities).flatten()
phi_recorded = fmodel_true.wind_directions
print("Initial power: ", power_recorded)
print("Initial rotor average velocities: ", U_recorded)


ti_recorded_new = 0.055
J_array, M_array, O_val, phi_predicted, ws_predicted, ti_predicted = fun.prediction(ti_recorded_new,wind_speed_recorded_new, wind_direction_recorded_new, fmodel_true, power_recorded, U_recorded, layout_x, layout_y, wind_shear,phi_recorded)


print(f'Il target era:')
print(f'wind speed real = {wind_speed_recorded}; wind direction real = {wind_direction_recorded}; ti real = {ti_recorded}')

wind_speeds = [ws_predicted]
ti_crit = [ti_predicted]
phi_crit = [phi_predicted]


fmodel_true.set(wind_speeds=wind_speeds, wind_directions=phi_crit, turbulence_intensities=ti_crit)
horizontal_plane = fmodel_true.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=150/200,
)
# Plot the flow field with rotors
fig, ax = plt.subplots()
visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    label_contours=False,
    title="Predicted",
)

# Plot the turbine rotors
layoutviz.plot_turbine_rotors(fmodel_true, ax=ax)
layoutviz.plot_turbine_labels(fmodel_true, ax=ax)

plt.show()
import warnings
warnings.filterwarnings('ignore')

# After the set method, the run method is called to perform the simulation
fmodel_true.run()

# There are functions to get either the power of each turbine, or the farm power
turbine_powers = fmodel_true.get_turbine_powers() / 1000.0
farm_power = fmodel_true.get_farm_power() / 1000.0

nt= len(layout_x)
nf= len(wind_speeds)   
print(f"The turbine power matrix should be of dimensions {nf} (n_findex) X {nt} (n_turbines)")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

print(f"The farm power should be a 1D array of length {nf} (n_findex)")
print(farm_power)
print("Shape: ", farm_power.shape)
import warnings
warnings.filterwarnings('ignore')

# Real Wind


fmodel_true.set(wind_speeds=wind_speed_recorded, wind_directions=wind_direction_recorded, turbulence_intensities=ti_recorded)
horizontal_plane = fmodel_true.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=150/200,
)
# Plot the flow field with rotors
fig, ax = plt.subplots()
visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    label_contours=False,
    title="Real",
)

# Plot the turbine rotors
layoutviz.plot_turbine_rotors(fmodel_true, ax=ax)
layoutviz.plot_turbine_labels(fmodel_true, ax=ax)

plt.show()
import warnings
warnings.filterwarnings('ignore')

# After the set method, the run method is called to perform the simulation
fmodel_true.run()

# There are functions to get either the power of each turbine, or the farm power
turbine_powers = fmodel_true.get_turbine_powers() / 1000.0
farm_power = fmodel_true.get_farm_power() / 1000.0

nt= len(layout_x)
nf= len(wind_speeds)   
print(f"The turbine power matrix should be of dimensions {nf} (n_findex) X {nt} (n_turbines)")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

print(f"The farm power should be a 1D array of length {nf} (n_findex)")
print(farm_power)
print("Shape: ", farm_power.shape)
import warnings
warnings.filterwarnings('ignore')