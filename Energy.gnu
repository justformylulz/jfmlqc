set terminal jpeg
set key
set output "Energies.jpeg"
set xlabel "Time / s"
set ylabel "Energy / eV"
plot 'Energy.txt' u ($0*0.005):1 w l title 'E(tot)', 'Energy.txt' u ($0*0.005):2 w l title 'E(P)', 'Energy.txt' u ($0*0.005):3 w l title 'E(kin)'
