import numpy as np


class GLCM:
    def __init__(self, image, step, num_features):
        self.matrix = image
        self.step = step
        self.co_occurrence_matrix = np.zeros((self.matrix.shape[0], self.matrix.shape[1]))
        self.co_occurrence_normalized = np.zeros((self.matrix.shape[0], self.matrix.shape[1]))
        self.features = np.zeros(num_features+1)
        
    def co_matrix_glcm(self):
        """
        The co occurence matrix with horizontal neighbor i.e. angle 0 [i][j + 1] horizontal
        """
        for i in range(0, self.matrix.shape[0]):
            for j in range(0, self.matrix.shape[1]-self.step,self.step):
                self.co_occurrence_matrix[int(self.matrix[i,j]), int(self.matrix[i,j+1])] += 1

        return self.co_occurrence_matrix
    
    def normalize_co_occurence(self):
        """
        Normalize the co occurence matrix with values between 0 and 1.
        """
        init_value, end_value = 0, 1
        self.co_occurrence_normalized = init_value + ((end_value*self.co_occurrence_matrix)/(self.matrix.shape[0]*(self.matrix.shape[1] -1)))
        
    def extract_features_from_glcm(self):
        """
        Extract nine features from GLCM
        """
        glcm = self.co_occurrence_matrix.shape[0]
        
        for i in range(glcm):
            for j in range(glcm):
                ij = self.co_occurrence_normalized[i,j]
                
                self.features[1] += ij*ij
                self.features[2] += ((i-j) * (i-j) * (ij))
                self.features[3] += (i*j) * ij
                self.features[5] += (ij)/(1+pow(i-j,2))
                self.features[9] += ij* np.log10(ij+ 1e-30)
                self.features[15] += (ij)/(1+abs(i-j))
                self.features[16] += ij*(i+j)
                self.features[21] += ij*abs(i-j)
                self.features[23] += ij*i*j
                self.features[24] += ij*pow(i-j,2)
                
        self.features[9] *= -1
        self.features[12] /= 2
        self.features[24] /= pow(pow(glcm, 2)-1,2)
        
        return self.features

