import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.units as u
import pandas as pd
import copy

###### AeroSandbox Setup

opti = asb.Opti()
# variable_categories_to_freeze = "all",
# freeze_style = "float"

make_plots=True

###### Lifting Surface

### Material
density_wing = 33.0 # kg/m^3 for depron foam

### Airfoil
airfoils = {
    name: asb.Airfoil(name=name,) for name in ["ag04", "naca0008"]
}

for v in airfoils.values():
   v.generate_polars(
       cache_filename = f"cache/{v.name}.json", alphas = np.linspace(-10, 10, 21)
    ) # generating aerodynamic polars using XFoil


##### Overall Specs

### Operating point
op_point = asb.OperatingPoint(
    velocity = opti.variable(init_guess = 14, lower_bound = 1, log_transform = True),
    alpha = opti.variable(init_guess = 0, lower_bound = -10, upper_bound = 10)
)

### Take off gross weight 
design_mass_TOGW = opti.variable(init_guess = 0.1, lower_bound = 1e-3)
design_mass_TOGW = np.maximum(design_mass_TOGW, 1e-3) # numerical clamp

### Cruise L/D
LD_cruise = opti.variable(init_guess = 15, lower_bound = 0.1, log_transform = True)

### Gravitational acceleration
g = 9.81

##### Vehicle Definition
# Datum (0, 0, 0) is coincident with the quarter-chord-point of the centerline cross section of the main wing

### Nose
x_nose = opti.variable(init_guess = -0.1, upper_bound = 1e-3,)

### Wing
wing_span = opti.variable(init_guess = 0.5, lower_bound = 0.3, upper_bound = 0.7)
wing_dihedral_angle_deg = opti.variable(init_guess = 11, lower_bound = 0, upper_bound = 20)
wing_root_chord = opti.variable(init_guess = 0.15, lower_bound = 1e-3,)
wing_taper = opti.variable(init_guess = 0.5, lower_bound = 0.3, upper_bound = 1)

def wing_rot(xyz):

    dihedral_rot = np.rotation_matrix_3D(angle = np.radians(wing_dihedral_angle_deg), axis = "X")

    return dihedral_rot @ np.array(xyz)

def wing_chord(y):

    half_span = wing_span / 2
    tip_chord = wing_taper * wing_root_chord
    spanfrac = np.abs(y) / half_span # 0 at root, 1 at tip

    return (1 - spanfrac) * wing_root_chord + spanfrac * tip_chord

def wing_twist(y):

    return np.zeros_like(y) # no twist

wing_ys = np.sinspace(0, wing_span / 2, 11, reverse_spacing = True) # y station

wing = asb.Wing(name = "Main Wing", symmetric = True,
    xsecs = [asb.WingXSec(xyz_le = wing_rot([-wing_chord(wing_ys[i]), wing_ys[i], 0.0]),
                          chord = wing_chord(wing_ys[i]),
                          airfoil = airfoils["ag04"],
                          twist = wing_twist(wing_ys[i]),
                          )
                          for i in range(np.length(wing_ys))
             ]      
).translate([0.75 * wing_root_chord, 0, 0])

### Horizontal Tailplane
AR_ht = 5.0
taper_ht = 0.7
l_ht = opti.variable(init_guess = 0.6, lower_bound = 0.2, upper_bound = 1.2)
S_ht = 0.12 * wing.area()
# S_ht = opti.variable(init_guess = 0.02, lower_bound = 1e-3,)

b_ht = np.sqrt(AR_ht * S_ht)
half_span_ht = b_ht / 2

def htail_chord(y):

    spanfrac = np.abs(y) / half_span_ht
    c_root_ht = 2 * S_ht / (b_ht * (1 + taper_ht))
    c_tip_ht  = taper_ht * c_root_ht

    return (1 - spanfrac) * c_root_ht + spanfrac * c_tip_ht

def htail_twist(y):

    return np.zeros_like(y) # no twist

htail_ys = np.sinspace(0, half_span_ht, 7, reverse_spacing=True) # y station

htail = asb.Wing(name = "HTail", symmetric = True,
    xsecs = [asb.WingXSec(xyz_le = [l_ht - htail_chord(htail_ys[i]), htail_ys[i], 0.0],
                          chord = htail_chord(htail_ys[i]),
                          twist = htail_twist(htail_ys[i]),
                          airfoil = airfoils["naca0008"],
                          )
                          for i in range(np.length(htail_ys))
             ]
)

### Vertical Tailplane
AR_vt = 2.0
taper_vt = 0.6
l_vt = l_ht
# l_vt = opti.variable(init_guess = 0.6, lower_bound = 0.2, upper_bound = 1.2)
S_vt = 0.06 * wing.area()
# S_vt = opti.variable(init_guess = 0.01, lower_bound = 1e-3,)

b_vt = np.sqrt(AR_vt * S_vt)

def vtail_chord(z):

    spanfrac = np.abs(z) / b_vt
    c_root_vt = 2 * S_vt / (b_vt * (1 + taper_vt))
    c_tip_vt  = taper_vt * c_root_vt
    return (1 - spanfrac) * c_root_vt + spanfrac * c_tip_vt

def vtail_twist(z):

    return np.zeros_like(z) # no twist

vtail_zs = np.sinspace(0, b_vt, 7, reverse_spacing=True) # z station

vtail = asb.Wing(name = "VTail", symmetric = False,
    xsecs = [asb.WingXSec(xyz_le = [l_vt - vtail_chord(vtail_zs[i]), 0.0, vtail_zs[i], ],
                          chord = vtail_chord(vtail_zs[i]),
                          twist = vtail_twist(vtail_zs[i]),
                          airfoil = airfoils["naca0008"],
                          )
                          for i in range(np.length(vtail_zs))
             ]
)

### Fuselage
x_tail = np.maximum(l_ht, l_vt)

fuselage = asb.Fuselage(name = "Fuse",
    xsecs = [asb.FuselageXSec(xyz_c = [x_nose, 0.0, 0.0], radius = 4e-3 / 2),
             asb.FuselageXSec(xyz_c=[x_tail, 0.0, 0.0], radius = 4e-3 / 2)
             ]
)

### Overall
airplane = asb.Airplane(
    name="Nausicaa",
    wings=[wing, htail, vtail],
    fuselages=[fuselage]
)


##### Internal Geometry and Weights
mass_props = {}

### Lifting surface centre of gravity
def lifting_surface_planform_cg(wing: asb.Wing, span_axis: str = "y"):

    # extract leading-edge positions and chords from xsecs
    xyz_le = np.stack([xsec.xyz_le for xsec in wing.xsecs], axis=0) # (N, 3)
    chords = np.array([xsec.chord for xsec in wing.xsecs]) # (N,)

    x_le = xyz_le[:, 0]
    y_le = xyz_le[:, 1]
    z_le = xyz_le[:, 2]

    if span_axis == "y":
        span = y_le
    elif span_axis == "z":
        span = z_le
    else:
        raise ValueError(f"span_axis must be 'y' or 'z', got {span_axis}")

    # spanwise strips between stations
    dspan = span[1:] - span[:-1] # strip width
    c_mid = 0.5 * (chords[:-1] + chords[1:]) # average chord

    # surface area
    A_strip_half = c_mid * dspan

    if wing.symmetric and span_axis == "y":
        A_strip = 2.0 * A_strip_half
    else:
        A_strip = A_strip_half

    # centroid x,z of each strip
    x_mid_i = x_le[:-1] + 0.5 * chords[:-1]
    x_mid_ip1 = x_le[1:] + 0.5 * chords[1:]
    x_mid_strip = 0.5 * (x_mid_i + x_mid_ip1)

    z_mid_strip = 0.5 * (z_le[:-1] + z_le[1:])

    A_total = np.sum(A_strip)

    x_cg = np.sum(A_strip * x_mid_strip) / A_total
    z_cg = np.sum(A_strip * z_mid_strip) / A_total

    return x_cg, z_cg

### Wing
x_cg_wing, z_cg_wing = lifting_surface_planform_cg(wing, span_axis="y")

mass_props['wing'] = asb.mass_properties_from_radius_of_gyration(
    mass = wing.volume() * density_wing,
    x_cg = x_cg_wing,
    # x_cg = (0.50 - 0.25) * wing_root_chord,
    z_cg = z_cg_wing,
    # z_cg = (0.03591) * (np.sind(wing_dihedral_angle_deg) / np.sind(11)) * (wing_span / 1),
    )

### Horizontal Tailplane
x_cg_ht, z_cg_ht = lifting_surface_planform_cg(htail, span_axis="y")

mass_props["htail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass = htail.volume() * density_wing,
    x_cg = x_cg_ht,
    # x_cg = htail.xsecs[0].xyz_le[0] + 0.50 * htail.xsecs[0].chord,
    z_cg = z_cg_ht,
)

### Vertical Tailplane
x_cg_vt, z_cg_vt = lifting_surface_planform_cg(vtail, span_axis="z")

mass_props["vtail_surfaces"] = asb.mass_properties_from_radius_of_gyration(
    mass = vtail.volume() * density_wing,
    x_cg = x_cg_vt,
    # x_cg=  vtail.xsecs[0].xyz_le[0] + 0.50 * vtail.xsecs[0].chord,
    z_cg = z_cg_vt,
)