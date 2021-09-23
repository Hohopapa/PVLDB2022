#!/bin/bash

./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 4 ../dat/real/itdk0304.A.0.00025 32 > time.multicuda.itdk.0.00025.4
./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 4 ../dat/real/itdk0304.A.0.0005 96 > time.multicuda.itdk.0.0005.4
./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 4 ../dat/real/itdk0304.A.0.001 192 > time.multicuda.itdk.0.001.4

./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 4 ../dat/real/twitter.A.0.00025 20 > time.multicuda.twitter.0.00025.4
./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 4 ../dat/real/twitter.A.0.0005 40 > time.multicuda.twitter.0.0005.4
./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 4 ../dat/real/twitter.A.0.001 81 > time.multicuda.twitter.0.001.4

./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 3 ../dat/real/itdk0304.A.0.00025 32 > time.multicuda.itdk.0.00025.3
./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 3 ../dat/real/itdk0304.A.0.0005 96 > time.multicuda.itdk.0.0005.3
./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 3 ../dat/real/itdk0304.A.0.001 192 > time.multicuda.itdk.0.001.3

./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 3 ../dat/real/twitter.A.0.00025 20 > time.multicuda.twitter.0.00025.3
./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 3 ../dat/real/twitter.A.0.0005 40 > time.multicuda.twitter.0.0005.3
./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 3 ../dat/real/twitter.A.0.001 81 > time.multicuda.twitter.0.001.3

./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 2 ../dat/real/itdk0304.A.0.00025 32 > time.multicuda.itdk.0.00025.2
./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 2 ../dat/real/itdk0304.A.0.0005 96 > time.multicuda.itdk.0.0005.2
./DeltaFastMultiCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 2 ../dat/real/itdk0304.A.0.001 192 > time.multicuda.itdk.0.001.2

./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 2 ../dat/real/twitter.A.0.00025 20 > time.multicuda.twitter.0.00025.2
./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 2 ../dat/real/twitter.A.0.0005 40 > time.multicuda.twitter.0.0005.2
./DeltaFastMultiCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 2 ../dat/real/twitter.A.0.001 81 > time.multicuda.twitter.0.001.2

