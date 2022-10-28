perfect_depth_rivs = []
great_depth_rivs = []
good_precip_rivs = []
good_temp_rivs = []

with open('eda_results/perfect_depth_rivs.txt', 'r') as in_file:
    perfect_depth_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/great_depth_rivs.txt', 'r') as in_file:
    great_depth_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/good_precip_rivs.txt', 'r') as in_file:
    good_precip_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/good_temp_rivs.txt', 'r') as in_file:
    good_temp_rivs = in_file.readline()[:-1].split(', ')

with open('eda_results/perfect_depth_rivs.txt', 'r') as in_file:
    perfect_depth_rivs = in_file.readline()[:-1].split(', ')

final_rivs = set(great_depth_rivs) & set(
    good_precip_rivs) & set(good_temp_rivs)
perfect_final_rivs = set(final_rivs) & set(perfect_depth_rivs)
imperfect_final_rivs = final_rivs - perfect_final_rivs

with open('eda_results/perfect_final_rivs.txt', 'w') as out:
    print(', '.join(perfect_final_rivs), file=out)
with open('eda_results/imperfect_final_rivs.txt', 'w') as out:
    print(', '.join(imperfect_final_rivs), file=out)
