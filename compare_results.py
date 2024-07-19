import os

results_path1 = '/workingspace/fhzny.proj/predict'
results_path2 = '/workingspace/fhzny.proj/deployment/build/results'

results1 = os.listdir(results_path1)
results2 = os.listdir(results_path2)
print(len(results1))
print(len(results2))
assert len(results1) == len(results2)
results1.sort()
results2.sort()

diff_num = 0
for i in range(len(results1)):
    with open(os.path.join(results_path1, results1[i]), 'r', encoding='utf-8') as reader:
        lines1 = reader.readlines()
    with open(os.path.join(results_path2, results2[i]), 'r', encoding='utf-8') as reader:
        lines2 = reader.readlines()
    
    # print(os.path.join(results_path1, results1[i]), os.path.join(results_path2, results2[i]))

    for j in range(min(len(lines1), len(lines2))):
        nums1 = [float(num) for num in lines1[j].strip().rstrip().split(' ')]
        nums2 = [float(num) for num in lines2[j].strip().rstrip().split(' ')]
        # print(len(nums1))
        assert len(nums1) == len(nums2)
        for k in range(len(nums1)):
                if not ((nums1[2] > 0.2) and (nums2[2] > 0.2)):
                    continue

                if (abs(nums1[k] - nums2[k]) > 20):
                        diff_num = diff_num + 1
                        print(i, j, k, " diff value:", abs(nums1[k] - nums2[k]))
                        print('nums1:', nums1)
                        print('nums2:', nums2)

print('Diff number:', diff_num)