set terminal png enhanced
set output "output/aggregated-weight.png"

set size square
set grid

set xrange [-10:10]
set yrange [-10:10]

plot './output/initial-weight.txt' title 'initial weight', './output/final-weight.txt' title 'final weight'

print "plotted image has been written in ./output/aggregated-weight.png"

