import math
from scipy import spatial

dataSetI = [3, 45]
dataSetII = [3, 46]
result = spatial.distance.cosine(dataSetI, dataSetII)
print(result)




def _make_vector(point_1: list, point_2: list):
        if len(point_1) != len(point_2): raise ValueError("Vectors has to be the same len!")
        vector = []
        for i in range(len(point_1)):
            vector.append(point_2[i] - point_1[i])
        return vector

def _get_angle_between_vectors(vector1, vector2):
    sum_of_coordinates = 0
    for i in range(len(vector1)):
        sum_of_coordinates += vector1[i] * vector2[i]
    return math.cos(
        sum_of_coordinates /
        (_get_vector_len(vector1) * _get_vector_len(vector2)))

def _get_vector_len(vector):
    sum_of_coordinates = 0
    for coordinate in vector:
        sum_of_coordinates += coordinate ** 2
    return math.sqrt(sum_of_coordinates)


a = 0, 6
b = 0, 10
c = 5, 5
#ab = _make_vector(a)
#ac = _make_vector(a)
bc = _make_vector(b, c)
#print('vectors', ab, ac, bc)
 
angle_a = _get_angle_between_vectors(a, b)
print(angle_a)




def _check_intersection(segment_one: list, segment_two: list) -> bool:
    if segment_two[0] <= segment_one[0] <= segment_two[1] or segment_two[0] <= segment_one[1] <= segment_two[1]:
        start = max(segment_two[0], segment_one[0])
        end = max(segment_two[1], segment_one[1])
        return [start, end]
    return False