set terminal png enhanced size 820, 820
set output "output/ridgelet-2d.png"

set size square
set pm3d map
set palette defined(-0.5 "blue", -0.2 "cyan", -0.1 "green", 0 "white", 0.18 "yellow", 0.5 "red")
set cbrange [-0.03:0.03]
set xlabel "a"
set ylabel "b"
set xtics ("-30" 0, "-20" 100, "-10" 200, "0" 300, "10" 400, "20" 500, "30" 599)
set ytics ("-30" 0, "-20" 100, "-10" 200, "0" 300, "10" 400, "20" 500, "30" 599)

splot './output/numerical-ridgelet.txt' matrix using 2:1:3

#splot './output/numerical-ridgelet.txt' using 1 : 2 : ($3 * exp(-($1 * -0.75 - $2)**2 / 2))

print "plotted image has been written in ./output/ridgelet-2d.png"

reset

set terminal png enhanced size 820, 820
set output "output/ridgelet-3d.png"

set size square
set pm3d at bs
unset surface
set palette defined(-0.5 "blue", -0.2 "cyan", -0.1 "green", 0 "white", 0.18 "yellow", 0.5 "red")
set cbrange [-0.03:0.03]
set xlabel "a"
set ylabel "b"
set xtics ("-30" 0, "-20" 100, "-10" 200, "0" 300, "10" 400, "20" 500, "30" 599)
set ytics ("-30" 0, "-20" 100, "-10" 200, "0" 300, "10" 400, "20" 500, "30" 599)

splot './output/numerical-ridgelet.txt' matrix using 2:1:3 notitle

print "plotted image has been written in ./output/ridgelet-3d.png"

reset

set terminal png enhanced size 800, 800
set output "output/reconstructed-function.png"

set size square
set grid
set samples 1000

eval "plot " . func . " with lines title 'original', './output/numerical-dual-ridgelet.txt' with lines title 'dual ridgelet'"

print "plotted image has been written in ./output/reconstructed-function.png"

