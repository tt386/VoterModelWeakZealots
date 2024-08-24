#!/bin/bash

path_dst=$1


Sources=(
	Simulations/Fig1_InitialObservation/SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.30000_Znum_1_minF_0.30000_maxF_0.30000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_2_LargestComponent_False_DataPoints_100000000/P_0.300_F_0.300/JustResults.png

	Simulations/Fig2_CompleteDistComparisons/fig.png

	Simulations/Fig3_SmallC/SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_4.000_Cnum_21_NodeNum_10000_minZ_0.01000_maxZ_0.75000_Znum_3_minF_1.00000_maxF_1.00000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_100_LargestComponent_False_DataPoints_100000000/SubC_Theory.png

	Simulations/Fig4_FullTheory/SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.90000_Znum_3_minF_0.30000_maxF_0.30000_Fnum_1_Timesteps_1000000000_SingleActive_False_Repeats_10_LargestComponent_False_DataPoints_100000000/SubC_Theory.png
	Simulations/Fig4_FullTheory/SaveFiles/Escape_Absorbing_ER_minC_0.000_maxC_10.000_Cnum_41_NodeNum_10000_minZ_0.30000_maxZ_0.30000_Znum_1_minF_0.30000_maxF_0.90000_Fnum_2_Timesteps_1000000000_SingleActive_False_Repeats_10_LargestComponent_False_DataPoints_100000000/SubC_Theory.png

)


Names=(
	Fig1_c.png

	Fig2.png

	Fig3_c.png

	Fig4_a.png
	Fig4_b.png
)





for i in "${!Sources[@]}"; do
    basename "${Sources[$i]}"
    f="${Names[$i]}"
    echo $filename
    file_dst="${path_dst}/${f}"
    
    echo $file_dst

    cp "${Sources[$i]}" "$file_dst"
    echo cp "${Sources[$i]}" "$file_dst"
done
