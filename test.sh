base_dir="" # define your base_dir here

run_test() {
    data_dir=$1
    full_name=$2
    load_size=$3
    num_test=$4
    dataroot="${base_dir}/${data_dir}"

    python test.py \
        --dataroot $dataroot \
        --results_dir results/test \
        --gpu_ids 0 \
        --name $full_name \
        --phase test \
        --preprocess scale_width \
        --load_size $load_size \
        --num_test $num_test \
}

# now we can call the function with the specific directory name and full name as the parameters

run_test "bdd100k_1_20" "bdd100k_1_20_tri_sem" 640 4025
# run_test "bdd100k_7_20_snowy" "bdd100k_7_20_snowy_tri_sem" 640 4025
# run_test "bdd100k_7_19_night" "bdd100k_7_19_night_tri_sem" 640 4025

# run_test "boreas_snowy" "boreas_snowy" 430 2089