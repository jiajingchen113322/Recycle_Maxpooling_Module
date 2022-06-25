import numpy as np


class accuracy_calculation():
    def __init__(self,confusion_matrix):
        self.confusion_matrix=confusion_matrix
        self.number_of_labels=self.confusion_matrix.shape[0]
    
    def get_over_all_accuracy(self):
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                all_values += self.confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += self.confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        return float(matrix_diagonal) / all_values

    def get_mean_class_accuracy(self):  # added
        re = 0
        for i in range(self.number_of_labels):
            re = re + self.confusion_matrix[i][i] / max(1,np.sum(self.confusion_matrix[i,:]))
        return re/self.number_of_labels

    def get_intersection_union_per_class(self):
        matrix_diagonal = [self.confusion_matrix[i][i] for i in range(self.number_of_labels)]
        errors_summed_by_row = [0] * self.number_of_labels
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_row[row] += self.confusion_matrix[row][column]
        errors_summed_by_column = [0] * self.number_of_labels
        for column in range(self.number_of_labels):
            for row in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_column[column] += self.confusion_matrix[row][column]

        divisor = [0] * self.number_of_labels
        for i in range(self.number_of_labels):
            divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
            if matrix_diagonal[i] == 0:
                divisor[i] = 1

        return [float(matrix_diagonal[i]) / divisor[i] for i in range(self.number_of_labels)]

    def get_average_intersection_union(self):
        values = self.get_intersection_union_per_class()
        class_seen = ((self.confusion_matrix.sum(1)+self.confusion_matrix.sum(0))!=0).sum()
        return sum(values) / class_seen


# if __name__=='__main__':
#     confusion_array=np.load('confusion_m.npy')
#     calcula_class=accuracy_calculation(confusion_array)
#     print(calcula_class.get_over_all_accuracy())
#     print(calcula_class.get_intersection_union_per_class())
#     print(calcula_class.get_average_intersection_union())
