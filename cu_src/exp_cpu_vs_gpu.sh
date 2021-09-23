#!/bin/bash

./DeltaFastCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.00025 48 > time.deltacuda.itdk.0.00025
./DeltaFastCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.0005 96 > time.deltacuda.itdk.0.0005
./DeltaFastCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.00075 144 > time.deltacuda.itdk.0.00075
./DeltaFastCuda ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.001 192 > time.deltacuda.itdk.0.001

./DeltaFast ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.00025 48 > time.deltafast.itdk.0.00025
./DeltaFast ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.0005 96 > time.deltafast.itdk.0.0005
./DeltaFast ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.00075 144 > time.deltafast.itdk.0.00075
./DeltaFast ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 ../dat/real/itdk0304.A.0.001 192 > time.deltafast.itdk.0.001

./DeltaFastCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.00025 20 > time.deltacuda.twitter.0.00025
./DeltaFastCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.0005 40 > time.deltacuda.twitter.0.0005
./DeltaFastCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.00075 60 > time.deltacuda.twitter.0.00075
./DeltaFastCuda ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.001 81 > time.deltacuda.twitter.0.001

./DeltaFast ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.00025 20 > time.deltafast.twitter.0.00025
./DeltaFast ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.0005 40 > time.deltafast.twitter.0.0005
./DeltaFast ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.00075 60 > time.deltafast.twitter.0.00075
./DeltaFast ../dat/real/twitter.V 81306 ../dat/real/twitter.E 1768149 ../dat/real/twitter.A.0.001 81 > time.deltafast.twitter.0.001
