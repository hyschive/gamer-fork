import yt
import numpy as np



#################################################################################
### Example Usage
'''
import yt
import add_ELBDM_derived_fields
ds = yt.load( './Data_000000' )
add_ELBDM_derived_fields.Add_ELBDM_derived_fields( ds )
sz = yt.SlicePlot( ds, 'z', ... )
'''
#################################################################################


#################################################################################
### Define the derived fields
#################################################################################

###################################
## Wavefunction
## psi = R + iI = f*e^(iS)
###################################

###################################
## Mass density
## rho = |psi|^2 = R^2 + I^2 = f^2
###################################

# m/hbar
def ELBDM_ETA(data):
   #return data.ds.parameters['ELBDM_Mass']*data.ds.units.code_mass/data.ds.units.reduced_planck_constant
   return data.ds.parameters['ELBDM_Mass']*data.ds.units.code_mass/(data.ds.parameters['ELBDM_PlanckConst']*data.ds.units.code_length**2*data.ds.units.code_mass/data.ds.units.code_time)

# f = sqrt(rho) = sqrt(R^2 + I^2)
def _f(field, data):
   return data["Dens"]**0.5

# R = sqrt(rho)*cos(S) = f*cos(S)
def _Real(field, data):
   return data["f"]*np.cos(data["Phase"])

# I = sqrt(rho)*sin(S) = f*sin(S)
def _Imag(field, data):
   return data["f"]*np.sin(data["Phase"])

# S = arctan(I/R)
def _S(field, data):
   return np.arctan2(data["Imag"], data["Real"])

# nu = 1/rho = 1/(R^2 + I^2)
def _specific_volume(field, data):
   return data["Dens"]**(-1)

#################################
## Momentum
#################################

# j_x = rho*v_x = (R*dI/dx - I*dR/dx)*hbar/m
def _bulk_momentum_density_x(field, data):
   return (data["Real"]*data["Imag_gradient_x"] - data["Imag"]*data["Real_gradient_x"])/ELBDM_ETA(data)

# j_y = rho*v_y = (R*dI/dy - I*dR/dy)*hbar/m
def _bulk_momentum_density_y(field, data):
   return (data["Real"]*data["Imag_gradient_y"] - data["Imag"]*data["Real_gradient_y"])/ELBDM_ETA(data)

# j_z = rho*v_z = (R*dI/dz - I*dR/dz)*hbar/m
def _bulk_momentum_density_z(field, data):
   return (data["Real"]*data["Imag_gradient_z"] - data["Imag"]*data["Real_gradient_z"])/ELBDM_ETA(data)

# |j| = rho*|v| = |R*grad(I) - I*grad(R)|*hbar/m
def _bulk_momentum_density_magnitude(field, data):
   return (data["bulk_momentum_density_x"]**2 + data["bulk_momentum_density_y"]**2 + data["bulk_momentum_density_z"]**2)**0.5

# xi_x = rho*w_x = (R*dR/dx + I*dI/dx)*hbar/m
def _thermal_momentum_density_x(field, data):
   return (data["Real"]*data["Real_gradient_x"] + data["Imag"]*data["Imag_gradient_x"])/ELBDM_ETA(data)

# xi_y = rho*w_y = (R*dR/dy + I*dI/dy)*hbar/m
def _thermal_momentum_density_y(field, data):
   return (data["Real"]*data["Real_gradient_y"] + data["Imag"]*data["Imag_gradient_y"])/ELBDM_ETA(data)

# xi_z = rho*w_z = (R*dR/dz + I*dI/dz)*hbar/m
def _thermal_momentum_density_z(field, data):
   return (data["Real"]*data["Real_gradient_z"] + data["Imag"]*data["Imag_gradient_z"])/ELBDM_ETA(data)

# |xi| = rho*|w| = |R*grad(R) + I*grad(I)|*hbar/m
def _thermal_momentum_density_magnitude(field, data):
   return (data["thermal_momentum_density_x"]**2 + data["thermal_momentum_density_y"]**2 + data["thermal_momentum_density_z"]**2)**0.5

# J_x = j_x*dV = dM*v_x
def _cell_bulk_momentum_x(field, data):
   return data["bulk_momentum_density_x"]*data["cell_volume"]

# J_y = j_y*dV = dM*v_y
def _cell_bulk_momentum_y(field, data):
   return data["bulk_momentum_density_y"]*data["cell_volume"]

# J_z = j_z*dV = dM*v_z
def _cell_bulk_momentum_z(field, data):
   return data["bulk_momentum_density_z"]*data["cell_volume"]

# |J| = |j|*dV = dM*|v|
def _cell_bulk_momentum_magnitude(field, data):
   #return (data["cell_bulk_momentum_x"]**2 + data["cell_bulk_momentum_y"]**2 + data["cell_bulk_momentum_z"]**2)**0.5
   return data["bulk_momentum_density_magnitude"]*data["cell_volume"]

# Xi_x = xi_x*dV = dM*w_x
def _cell_thermal_momentum_x(field, data):
   return data["thermal_momentum_density_x"]*data["cell_volume"]

# Xi_y = xi_y*dV = dM*w_y
def _cell_thermal_momentum_y(field, data):
   return data["thermal_momentum_density_y"]*data["cell_volume"]

# Xi_z = xi_z*dV = dM*w_z
def _cell_thermal_momentum_z(field, data):
   return data["thermal_momentum_density_z"]*data["cell_volume"]

# |Xi| = |xi|*dV = dM*|w|
def _cell_thermal_momentum_magnitude(field, data):
   #return (data["cell_thermal_momentum_x"]**2 + data["cell_thermal_momentum_y"]**2 + data["cell_thermal_momentum_z"]**2)**0.5
   return data["thermal_momentum_density_magnitude"]*data["cell_volume"]

#################################
## Energy
#################################

# e_k_bulk = 1/2*rho*|v|^2 = 1/2*rho*|grad(S)|^2*hbar^2/m^2
def _bulk_kinetic_energy_density(field, data):
   #return 0.5*data["Dens"]*data["bulk_velocity"]**2
   return 0.5*(data["bulk_momentum_density_magnitude"]**2)/data["Dens"]

# e_k_thml = 1/2*rho*|w|^2 = 1/2*|grad(f)|^2*hbar^2/m^2
def _thermal_kinetic_energy_density(field, data):
   #return 0.5*data["Dens"]*data["thermal_velocity"]**2
   #return 0.5*(data["f_gradient_magnitude"]**2)/ELBDM_ETA(data)**2
   return 0.5*(data["thermal_momentum_density_magnitude"]**2)/data["Dens"]

# e_k = 1/2*rho*|u|^2 = 1/2*rho*|v|^2 + 1/2*rho*|w|^2 = 1/2*(|grad(R)|^2 + |grad(I)|^2)*hbar^2/m^2
def _total_kinetic_energy_density(field, data):
   #return data["bulk_kinetic_energy_density"] + data["thermal_kinetic_energy_density"]
   return 0.5*(data["Real_gradient_magnitude"]**2 + data["Imag_gradient_magnitude"]**2)/ELBDM_ETA(data)**2

# E_k_bulk = e_k_bulk*dV
def _cell_bulk_kinetic_energy(field, data):
   return data["bulk_kinetic_energy_density"]*data["cell_volume"]

# E_k_thml = e_k_thml*dV
def _cell_thermal_kinetic_energy(field, data):
   return data["thermal_kinetic_energy_density"]*data["cell_volume"]

# E_k = e_k*dV
def _cell_total_kinetic_energy(field, data):
   return data["total_kinetic_energy_density"]*data["cell_volume"]

# epsilon_k = 1/2*u^2 = 1/2*v^2 +1/2*w^2
def _specific_total_kinetic_energy(field, data):
   return data["total_kinetic_energy_density"]*data["specific_volume"]

# e_p = 1/2*rho*Phi
def _potential_energy_density(field, data):
   return 0.5*data['Pote']*data['Dens']

# E_p = e_p*dV
def _cell_potential_energy(field, data):
   return data["potential_energy_density"]*data["cell_volume"]

# epsilon_p = Phi_0 - Phi
def _relative_potential(field, data):
   Phi_0 = 0.0
   return Phi_0 - data['Pote']

# e = e_p + e_k
def _total_energy_density(field, data):
   return data['potential_energy_density'] + data['total_kinetic_energy_density']

# E = E_p + E_k
def _cell_total_energy(field, data):
   return data['cell_potential_energy'] + data['cell_total_kinetic_energy']

# epsilon = epsilon_p + epsilon_k = Phi_0 - Phi - 1/2*u^2
def _relative_energy(field, data):
   return data['relative_potential'] - data["specific_total_kinetic_energy"]

#################################
## Velocity
#################################

# v_x = (dS/dx)*hbar/m = (R*dI/dx - I*dR/dx)/rho*hbar/m
def _bulk_velocity_x(field, data):
   return data["bulk_momentum_density_x"]/data["Dens"]

# v_y = (dS/dy)*hbar/m = (R*dI/dy - I*dR/dy)/rho*hbar/m
def _bulk_velocity_y(field, data):
   return data["bulk_momentum_density_y"]/data["Dens"]

# v_z = (dS/dz)*hbar/m = (R*dI/dz - I*dR/dz)/rho*hbar/m
def _bulk_velocity_z(field, data):
   return data["bulk_momentum_density_z"]/data["Dens"]

# |v| = |grad(S)|*hbar/m = |(R*grad(I) - I*grad(R))|/rho*hbar/m
def _bulk_velocity_magnitude(field, data):
   #return (data["bulk_velocity_x"]**2 + data["bulk_velocity_y"]**2 + data["bulk_velocity_z"]**2)**0.5
   return data["bulk_momentum_density_magnitude"]/data["Dens"]

# w_x = ((df/dx)/f)*hbar/m = (R*dR/dx + I*dI/dx)/rho*hbar/m
def _thermal_velocity_x(field, data):
   #return (data["f_gradient_x"]/data["f"])/ELBDM_ETA(data)
   return data["thermal_momentum_density_x"]/data["Dens"]

# w_y = ((df/dy)/f)*hbar/m = (R*dR/dy + I*dI/dy)/rho*hbar/m
def _thermal_velocity_y(field, data):
   #return (data["f_gradient_y"]/data["f"])/ELBDM_ETA(data)
   return data["thermal_momentum_density_y"]/data["Dens"]

# w_z = ((df/dz)/f)*hbar/m = (R*dR/dz + I*dI/dz)/rho*hbar/m
def _thermal_velocity_z(field, data):
   #return (data["f_gradient_z"]/data["f"])/ELBDM_ETA(data)
   return data["thermal_momentum_density_z"]/data["Dens"]

# |w| = |(grad(f)/f)|*hbar/m = |(R*grad(R) + I*grad(I))|/rho*hbar/m
def _thermal_velocity_magnitude(field, data):
   #return (data["f_gradient_magnitude"]/data["f"])/ELBDM_ETA(data)
   return data["thermal_momentum_density_magnitude"]/data["Dens"]

# |u| = sqrt(|grad(f)/f|^2 + |grad(S)|^2)*hbar/m = sqrt((|grad(R)|^2 + |grad(I)|^2)/rho)*hbar/m
def _total_velocity_magnitude(field, data):
   #return (data["bulk_velocity_magnitude"]**2 + data["thermal_velocity_magnitude"]**2)**0.5
   #return (((data["Real_gradient_magnitude"]**2 + data["Imag_gradient_magnitude"]**2)/data["Dens"])**0.5)/ELBDM_ETA(data)
   return (2.0*data["specific_total_kinetic_energy"])**0.5

#################################
## Wavevector
#################################

# k_bulk_x = dS/dx = (R*dI/dx - I*dR/dx)/rho
def _bulk_wavevector_x(field, data):
   return data["bulk_velocity_x"]*ELBDM_ETA(data)

# k_bulk_y = dS/dy = (R*dI/dy - I*dR/dy)/rho
def _bulk_wavevector_y(field, data):
   return data["bulk_velocity_y"]*ELBDM_ETA(data)

# k_bulk_z = dS/dz = (R*dI/dz - I*dR/dz)/rho
def _bulk_wavevector_z(field, data):
   return data["bulk_velocity_z"]*ELBDM_ETA(data)

# |k_bulk| = |grad(S)| = |(R*grad(I) - I*grad(R))|/rho
def _bulk_wavevector_magnitude(field, data):
   return data["bulk_velocity_magnitude"]*ELBDM_ETA(data)

# k_thml_x = (df/dx)/f = (R*dR/dx + I*dI/dx)/rho
def _thermal_wavevector_x(field, data):
   return data["thermal_velocity_x"]*ELBDM_ETA(data)

# k_thml_y = (df/dy)/f = (R*dR/dy + I*dI/dy)/rho
def _thermal_wavevector_y(field, data):
   return data["thermal_velocity_y"]*ELBDM_ETA(data)

# k_thml_z = (df/dz)/f = (R*dR/dz + I*dI/dz)/rho
def _thermal_wavevector_z(field, data):
   return data["thermal_velocity_z"]*ELBDM_ETA(data)

# |k_thml| = |grad(f)/f| = |(R*grad(R) + I*grad(I))|/rho
def _thermal_wavevector_magnitude(field, data):
   return data["thermal_velocity_magnitude"]*ELBDM_ETA(data)

# |k| = sqrt(|grad(f)/f|^2 + |grad(S)|^2) = sqrt((|grad(R)|^2 + |grad(I)|^2)/rho)
def _total_wavevector_magnitude(field, data):
   #return (data["bulk_wavevector_magnitude"]**2 + data["thermal_wavevector_magnitude"]**2)**0.5
   #return ((data["Real_gradient_magnitude"]**2 + data["Imag_gradient_magnitude"]**2)/data["Dens"])**0.5
   return data["total_velocity_magnitude"]*ELBDM_ETA(data)

#################################
# Wavelength
#################################

# lambda_bulk_x = 2*PI/k_bulk_x
def _bulk_wavelength_x(field, data):
   return 2.0*np.pi/data["bulk_wavevector_x"]

# lambda_bulk_y = 2*PI/k_bulk_y
def _bulk_wavelength_y(field, data):
   return 2.0*np.pi/data["bulk_wavevector_y"]

# lambda_bulk_z = 2*PI/k_bulk_z
def _bulk_wavelength_z(field, data):
   return 2.0*np.pi/data["bulk_wavevector_z"]

# lambda_bulk = 2*PI/k_bulk = 2*PI*hbar/(m*v)
def _bulk_wavelength(field, data):
   return 2.0*np.pi/data["bulk_wavevector_magnitude"]

# lambda_thml_x = 2*PI/k_thml_x
def _thermal_wavelength_x(field, data):
   return 2.0*np.pi/data["thermal_wavevector_x"]

# lambda_thml_y = 2*PI/k_thml_y
def _thermal_wavelength_y(field, data):
   return 2.0*np.pi/data["thermal_wavevector_y"]

# lambda_thml_z = 2*PI/k_thml_z
def _thermal_wavelength_z(field, data):
   return 2.0*np.pi/data["thermal_wavevector_z"]

# lambda_thml = 2*PI/k_thml = 2*PI*hbar/(m*w)
def _thermal_wavelength(field, data):
   return 2.0*np.pi/data["thermal_wavevector_magnitude"]

# lambda = 2*PI/k = 2*PI*hbar/(m*u)
def _total_wavelength(field, data):
   return 2.0*np.pi/data["total_wavevector_magnitude"]

# lambda_bulk/dx = 2*PI/(k_bulk*dx)
def _cells_per_bulk_wavelength(field, data):
   return data["bulk_wavelength"]/data["dx"]

# lambda_thml/dx = 2*PI/(k_thml*dx)
def _cells_per_thermal_wavelength(field, data):
   return data["thermal_wavelength"]/data["dx"]

# lambda/dx = 2*PI/(k*dx)
def _cells_per_total_wavelength(field, data):
   return data["total_wavelength"]/data["dx"]

# dx/lambda_bulk = k_bulk*dx/(2*PI)
def _bulk_wavelength_per_cell(field, data):
   return 1.0/data["cells_per_bulk_wavelength"]

# dx/lambda_thml = k_thml*dx/(2*PI)
def _thermal_wavelength_per_cell(field, data):
   return 1.0/data["cells_per_thermal_wavelength"]

# dx/lambda = k*dx/(2*PI)
def _total_wavelength_per_cell(field, data):
   return 1.0/data["cells_per_total_wavelength"]

#################################
## Second Derivative
#################################

# Laplacian(f) = div(grad(f))
def _f_laplacian(field, data):
   return data["f_gradient_x_gradient_x"] + data["f_gradient_y_gradient_y"] + data["f_gradient_z_gradient_z"]

# Laplacian(S) = div(grad(S)) = div(v)*m/hbar = div(k_bulk)
def _S_laplacian(field, data):
   return data["bulk_wavevector_x_gradient_x"] + data["bulk_wavevector_y_gradient_y"] + data["bulk_wavevector_z_gradient_z"]

# Q = -1/2*(Laplacian(f)/f)*hbar^2/m^2
def _quantum_pressure_potential(field, data):
   return -0.5*(data["f_laplacian"]/data["f"])/ELBDM_ETA(data)**2


#################################################################################
### Add the derived fields
#################################################################################

def Add_ELBDM_derived_fields(ds):

   ## Wavefunction
   ds.add_field(       ("gamer", "f"),
                 function      = _f,
                 display_name  =r"$f$",
                 units         = "code_mass**0.5/code_length**1.5",
                 sampling_type = "cell" )

   if ds.parameters["ELBDMScheme"] == 2: # ELBDM_HYBRID
      ds.add_field(    ("gamer", "Real"),
                 function      = _Real,
                 display_name  =r"Real",
                 units         = "code_mass**0.5/code_length**1.5",
                 sampling_type = "cell" )

      ds.add_field(    ("gamer", "Imag"),
                 function      = _Imag,
                 display_name  =r"Imag",
                 units         = "code_mass**0.5/code_length**1.5",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "S"),
                 function      = _S,
                 display_name  =r"$S$",
                 units         = "dimensionless",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "specific_volume"),
                 function      = _specific_volume,
                 display_name  =r"Specific Volume $\frac{1}{\rho}$",
                 units         = "code_length**3/code_mass",
                 sampling_type = "cell" )

   ## Gradient Field
   Grad_R = ds.add_gradient_fields( ("gamer","Real") )
   Grad_I = ds.add_gradient_fields( ("gamer","Imag") )
   Grad_f = ds.add_gradient_fields( ("gamer","f")    )
   Grad_S = ds.add_gradient_fields( ("gamer","S")    )

   ## Momentum
   ds.add_field(       ("gamer", "bulk_momentum_density_x"),
                 function      = _bulk_momentum_density_x,
                 display_name  =r"Bulk Momentum Density X $j_x$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_momentum_density_y"),
                 function      = _bulk_momentum_density_y,
                 display_name  =r"Bulk Momentum Density Y $j_y$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_momentum_density_z"),
                 function      = _bulk_momentum_density_z,
                 display_name  =r"Bulk Momentum Density Z $j_z$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_momentum_density_magnitude"),
                 function      = _bulk_momentum_density_magnitude,
                 display_name  =r"Bulk Momentum Density Magnitude $|\vec{j}|$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "thermal_momentum_density_x"),
                 function      = _thermal_momentum_density_x,
                 display_name  =r"Thermal Momentum Density X $\xi_x$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_momentum_density_y"),
                 function      = _thermal_momentum_density_y,
                 display_name  =r"Thermal Momentum Density Y $\xi_y$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_momentum_density_z"),
                 function      = _thermal_momentum_density_z,
                 display_name  =r"Thermal Momentum Density Z $\xi_z$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_momentum_density_magnitude"),
                 function      = _thermal_momentum_density_magnitude,
                 display_name  =r"Thermal Momentum Density Magnitude $|\vec{\xi}|$",
                 units         = "code_mass/(code_length**2*code_time)",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_bulk_momentum_x"),
                 function      = _cell_bulk_momentum_x,
                 display_name  =r"Cell Bulk Momentum X $J_x$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_bulk_momentum_y"),
                 function      = _cell_bulk_momentum_y,
                 display_name  =r"Cell Bulk Momentum Y $J_y$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_bulk_momentum_z"),
                 function      = _cell_bulk_momentum_z,
                 display_name  =r"Cell Bulk Momentum Z $J_z$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_bulk_momentum_magnitude"),
                 function      = _cell_bulk_momentum_magnitude,
                 display_name  =r"Cell Bulk Momentum Magnitude $|\vec{J}|$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_thermal_momentum_x"),
                 function      = _cell_thermal_momentum_x,
                 display_name  =r"Cell Thermal Momentum X $\Xi_x$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_thermal_momentum_y"),
                 function      = _cell_thermal_momentum_y,
                 display_name  =r"Cell Thermal Momentum Y $\Xi_y$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_thermal_momentum_z"),
                 function      = _cell_thermal_momentum_z,
                 display_name  =r"Cell Thermal Momentum Z $\Xi_z$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ds.add_field(       ("gamer", "cell_thermal_momentum_magnitude"),
                 function      = _cell_thermal_momentum_magnitude,
                 display_name  =r"Cell Thermal Momentum Magnitude $|\vec{\Xi}|$",
                 units         = "code_mass*code_length/code_time",
                 sampling_type = "cell")

   ## Energy
   ds.add_field(       ("gamer", "bulk_kinetic_energy_density"),
                 function      = _bulk_kinetic_energy_density,
                 display_name  =r"Bulk Kinetic Energy Density",
                 units         = "code_mass/(code_length*code_time**2)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_kinetic_energy_density"),
                 function      = _thermal_kinetic_energy_density,
                 display_name  =r"Thermal Kinetic Energy Density",
                 units         = "code_mass/(code_length*code_time**2)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "total_kinetic_energy_density"),
                 function      = _total_kinetic_energy_density,
                 display_name  =r"Total Kinetic Energy Density",
                 units         = "code_mass/(code_length*code_time**2)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "cell_bulk_kinetic_energy"),
                 function      = _cell_bulk_kinetic_energy,
                 display_name  =r"Cell Bulk Kinetic Energy $E_{\rm{k,bulk}}$",
                 units         = "code_mass*code_length**2/(code_time**2)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "cell_thermal_kinetic_energy"),
                 function      = _cell_thermal_kinetic_energy,
                 display_name  =r"Cell Thermal Kinetic Energy $E_{\rm{k,thermal}}$",
                 units         = "code_mass*code_length**2/(code_time**2)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "cell_total_kinetic_energy"),
                 function      = _cell_total_kinetic_energy,
                 display_name  =r"Cell Total Kinetic Energy $E_{\rm{k}}$",
                 units         = "code_mass*code_length**2/(code_time**2)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "specific_total_kinetic_energy"),
                 function      = _specific_total_kinetic_energy,
                 display_name  =r"Specific Total Kinetic Energy $\epsilon_{\rm{k}}$",
                 units         = "code_length**2/(code_time**2)",
                 sampling_type = "cell" )

   if ds.parameters["Gravity"] == 1:
      if ds.parameters["Opt__Output_Pot"] == 1:

         ds.add_field( ("gamer", "potential_energy_density"),
                 function      = _potential_energy_density,
                 display_name  =r"Potential Energy Density",
                 units         = "code_mass/(code_length*code_time**2)",
                 sampling_type = "cell" )

         ds.add_field( ("gamer", "cell_potential_energy"),
                 function      = _cell_potential_energy,
                 display_name  =r"Cell Potential Energy $E_{\rm{p}}$",
                 units         = "code_mass*code_length**2/(code_time**2)",
                 sampling_type = "cell" )

         ds.add_field( ("gamer", "relative_potential"),
                 function      = _relative_potential,
                 display_name  =r"Relative Potential $\epsilon_{\rm{p}}$",
                 units         = "code_length**2/(code_time**2)",
                 sampling_type = "cell" )

         ds.add_field( ("gamer", "total_energy_density"),
                 function      = _total_energy_density,
                 display_name  =r"Total Energy Density",
                 units         = "code_mass/(code_length*code_time**2)",
                 sampling_type = "cell" )

         ds.add_field( ("gamer", "cell_total_energy"),
                 function      = _cell_total_energy,
                 display_name  =r"Cell Total Energy $E$",
                 units         = "code_mass*code_length**2/(code_time**2)",
                 sampling_type = "cell" )

         ds.add_field( ("gamer", "relative_energy"),
                 function      = _relative_energy,
                 display_name  =r"Relative Energy $\epsilon$",
                 units         = "code_length**2/(code_time**2)",
                 sampling_type = "cell" )

   ## Velocity
   ds.add_field(       ("gamer", "bulk_velocity_x"),
                 function      = _bulk_velocity_x,
                 display_name  =r"Bulk Velocity X $v_x$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_velocity_y"),
                 function      = _bulk_velocity_y,
                 display_name  =r"Bulk Velocity Y $v_y$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_velocity_z"),
                 function      = _bulk_velocity_z,
                 display_name  =r"Bulk Velocity Z $v_z$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_velocity_magnitude"),
                 function      = _bulk_velocity_magnitude,
                 display_name  =r"Bulk Velocity Magnitude $|\vec{v}|$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_velocity_x"),
                 function      = _thermal_velocity_x,
                 display_name  =r"Thermal Velocity X $w_x$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_velocity_y"),
                 function      = _thermal_velocity_y,
                 display_name  =r"Thermal Velocity Y $w_y$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_velocity_z"),
                 function      = _thermal_velocity_z,
                 display_name  =r"Thermal Velocity Z $w_z$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_velocity_magnitude"),
                 function      = _thermal_velocity_magnitude,
                 display_name  =r"Thermal Velocity Magnitude $|\vec{w}|$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "total_velocity_magnitude"),
                 function      = _total_velocity_magnitude,
                 display_name  =r"Total Velocity Magnitude $|\vec{u}|$",
                 units         = "code_length/code_time",
                 sampling_type = "cell" )

   ## Wavevector
   ds.add_field(       ("gamer", "bulk_wavevector_x"),
                 function      = _bulk_wavevector_x,
                 display_name  =r"Bulk Wavevector X $k_{\rm{bulk},x}$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_wavevector_y"),
                 function      = _bulk_wavevector_y,
                 display_name  =r"Bulk Wavevector Y $k_{\rm{bulk},y}$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_wavevector_z"),
                 function      = _bulk_wavevector_z,
                 display_name  =r"Bulk Wavevector Z $k_{\rm{bulk},z}$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_wavevector_magnitude"),
                 function      = _bulk_wavevector_magnitude,
                 display_name  =r"Bulk Wavevector Magnitude $|\vec{k}_{\rm{bulk}}|$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavevector_x"),
                 function      = _thermal_wavevector_x,
                 display_name  =r"Thermal Wavevector X $k_{\rm{thermal},x}$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavevector_y"),
                 function      = _thermal_wavevector_y,
                 display_name  =r"Thermal Wavevector Y $k_{\rm{thermal},y}$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavevector_z"),
                 function      = _thermal_wavevector_z,
                 display_name  =r"Thermal Wavevector Z $k_{\rm{thermal},z}$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavevector_magnitude"),
                 function      = _thermal_wavevector_magnitude,
                 display_name  =r"Thermal Wavevector Magnitude $|\vec{k}_{\rm{thermal}}|$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "total_wavevector_magnitude"),
                 function      = _total_wavevector_magnitude,
                 display_name  =r"Total Wavevector Magnitude $|\vec{k}|$",
                 units         = "1/(code_length)",
                 sampling_type = "cell" )

   ## Wavelength
   ds.add_field(       ("gamer", "bulk_wavelength_x"),
                 function      = _bulk_wavelength_x,
                 display_name  =r"Bulk Wavelength X $\lambda_{\rm{bulk},x}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_wavelength_y"),
                 function      = _bulk_wavelength_y,
                 display_name  =r"Bulk Wavelength Y $\lambda_{\rm{bulk},y}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_wavelength_z"),
                 function      = _bulk_wavelength_z,
                 display_name  =r"Bulk Wavelength Z $\lambda_{\rm{bulk},z}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_wavelength"),
                 function      = _bulk_wavelength,
                 display_name  =r"Bulk Wavelength $\lambda_{\rm{bulk}}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavelength_x"),
                 function      = _thermal_wavelength_x,
                 display_name  =r"Thermal Wavelength X $\lambda_{\rm{thermal},x}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavelength_y"),
                 function      = _thermal_wavelength_y,
                 display_name  =r"Thermal Wavelength Y $\lambda_{\rm{thermal},y}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavelength_z"),
                 function      = _thermal_wavelength_z,
                 display_name  =r"Thermal Wavelength Z $\lambda_{\rm{thermal},z}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavelength"),
                 function      = _thermal_wavelength,
                 display_name  =r"Thermal Wavelength $\lambda_{\rm{thermal}}$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "total_wavelength"),
                 function      = _total_wavelength,
                 display_name  =r"Total Wavelength $\lambda$",
                 units         = "code_length",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "cells_per_bulk_wavelength"),
                 function      = _cells_per_bulk_wavelength,
                 display_name  =r"$\lambda_{\rm{bulk}} / \Delta x$",
                 units         = "dimensionless",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "cells_per_thermal_wavelength"),
                 function      = _cells_per_thermal_wavelength,
                 display_name  =r"$\lambda_{\rm{thermal}} / \Delta x$",
                 units         = "dimensionless",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "cells_per_total_wavelength"),
                 function      = _cells_per_total_wavelength,
                 display_name  =r"$\lambda / \Delta x$",
                 units         = "dimensionless",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "bulk_wavelength_per_cell"),
                 function      = _bulk_wavelength_per_cell,
                 display_name  =r"$\Delta x / \lambda_{\rm{bulk}}$",
                 units         = "dimensionless",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "thermal_wavelength_per_cell"),
                 function      = _thermal_wavelength_per_cell,
                 display_name  =r"$\Delta x / \lambda_{\rm{thermal}}$",
                 units         = "dimensionless",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "total_wavelength_per_cell"),
                 function      = _total_wavelength_per_cell,
                 display_name  =r"$\Delta x / \lambda$",
                 units         = "dimensionless",
                 sampling_type = "cell" )

   ## Second Derivative
   Grad_f_gradient_x = ds.add_gradient_fields( ("gamer","f_gradient_x")      )
   Grad_f_gradient_y = ds.add_gradient_fields( ("gamer","f_gradient_y")      )
   Grad_f_gradient_z = ds.add_gradient_fields( ("gamer","f_gradient_z")      )
   Grad_S_gradient_x = ds.add_gradient_fields( ("gamer","bulk_wavevector_x") )
   Grad_S_gradient_y = ds.add_gradient_fields( ("gamer","bulk_wavevector_y") )
   Grad_S_gradient_z = ds.add_gradient_fields( ("gamer","bulk_wavevector_z") )

   ds.add_field(       ("gamer", "f_laplacian"),
                 function      = _f_laplacian,
                 display_name  =r"$\nabla^2 f$",
                 units         = "code_mass**0.5/code_length**3.5",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "S_laplacian"),
                 function      = _S_laplacian,
                 display_name  =r"$\nabla^2 S$",
                 units         = "1/code_length**2",
                 sampling_type = "cell" )

   ds.add_field(       ("gamer", "quantum_pressure_potential"),
                 function      = _quantum_pressure_potential,
                 display_name  =r"Quantum Pressure Potential $Q$",
                 units         = "code_length**2/code_time**2",
                 sampling_type = "cell" )
