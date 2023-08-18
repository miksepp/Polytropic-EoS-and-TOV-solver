# Polytropic-EoS-and-TOV-solver
A simple code to generate random polytropic equation of states for neutron stars and solve the Tolman-Oppenheimer-Volkoff equations. 

The generated EoSs are constrained to be consistent with astrophysical data, namely neutron star mass measurements and tidal deformability measurements. EoSs not satisfying the mass constraint are plotted in cyan, while those not satisfying the tidal constraint are shown in magenta. Gren EoSs pass both constraint and are thus possible.

The code is given in Python and Julia, and although they produce the same results, the Python version is incomplete and missing some features.
