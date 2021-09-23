#!/usr/bin/gnuplot

set term postscript eps 26
set output 'stackedhisto1.eps'

# Where to put the legend
# and what it should contain
set size 1.5,1
set multiplot

set origin 0,0
set size 0.65,1

#set key invert reverse Left outside
#set key autotitle columnheader
unset key
set yrange [0:6900]
set xrange [0:3]
set ylabel "Execution time"

# Define plot style 'stacked histogram'
# with additional settings
set style data histogram
set style histogram rowstacked
set style fill solid border -1
set boxwidth 0.75


# We are plotting columns 2, 3 and 4 as y-values,
# the x-ticks are coming from column 1
# Additionally to the graph above we need to specify
# the titles via 't 2' aso.
plot 'stackedhisto.dat' using 2:xtic(1) fs pattern 4 \
    ,'' using 3 fs pattern 2 \
    ,'' using 4 fs pattern 5 \
    ,'' using 5 fs pattern 6 \
    ,'' using 6 fs pattern 7

#reset
#set term postscript eps 26
#set output 'stackedhisto2.eps'

# Where to put the legend
# and what it should contain
set origin 0.65,0
set size 0.85,1

set key invert reverse Left outside
set key autotitle columnheader

set yrange [0:100]
set xrange [-1:2]
set ylabel "Percentage of computation"

# Define plot style 'stacked histogram'
# with additional settings
set style data histogram
set style histogram rowstacked
set style fill solid border -1
set boxwidth 0.75


# We are plotting columns 2, 3 and 4 as y-values,
# the x-ticks are coming from column 1
# Additionally to the graph above we need to specify
# the titles via 't 2' aso.
plot 'stackedhisto.dat' using (100*$2/($2+$3+$4+$5+$6)):xtic(1) t column(2) fs pattern 4 \
    ,'' using (100*$3/($2+$3+$4+$5+$6)) t column(3) fs pattern 2 \
    ,'' using (100*$4/($2+$3+$4+$5+$6)) t column(4) fs pattern 5 \
    ,'' using (100*$5/($2+$3+$4+$5+$6)) t column(5) fs pattern 6 \
    ,'' using (100*$6/($2+$3+$4+$5+$6)) t column(6) fs pattern 7


unset multiplot
