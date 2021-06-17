set terminal jpeg
set key
set output "Energies.jpeg"
set xlabel "Timestep"
set ylabel "Energy"
plot 'Energy.txt' u 1 w l title 'E(tot)', 'Energy.txt' u 2 w l title 'E(P)', 'Energy.txt' u 3 w l title 'E(kin)'
