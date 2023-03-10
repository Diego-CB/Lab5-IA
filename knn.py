# Universidad del Valle de Guatemala
# Inteligencia artificial
# Cristian Aguirre
# Diego Córdova
# Paola De León

# Clase para simular librería de sklearn de un KNN.

from collections import Counter
import numpy as np


class KNN:

    def __init__(self, k=3):
        """
            K-Nearest Neighbors Algorithm.
            
            Parámetros:
            -----------
            - k: Número de vecinos. (Valor default: 3)
        """
        self.k = k

    def fit(self, X, y):
        """
            Ajustar el clasificador del algoritmo con la data training.
           
            Parámetros:
            -----------
            - X: Variables dependientes.
            - y: Variable independiente.
        """
        self.X_entreno = X
        self.y_entreno = y

    def euclideanDistance(self, x1, x2):
        """
            Cálculo de distancia Euclidiana.
            
            Parámetros:
            -----------
            - x1 y x2: Array.

            Returns:
            --------
            Distancia euclidiana calculada.
        """
        x1 = np.array(x1)
        x2 = np.array(x2)

        return np.sum((x1 - x2) ** 2)

    def kDistance (self, x):
        """
            Cálculo y elección de la clase a la qué pertenece el punto.

            Parámetros:
            - x: 
        """
        # Obtener las distancias euclidianas y ordenarlas
        euclideanDistances = []
        for x_entreno in self.X_entreno:
            euclideanDistances.append(self.euclideanDistance(x, x_entreno))

        kValues = np.argsort(euclideanDistances)[:self.k]

        # Votación
        masCernana = []
        for i in kValues:
            masCernana.append(self.y_entreno[i])

        # Obtener clasificación con más elementos (el más común)
        comun = Counter(masCernana)
        comun = comun.most_common(1)

        return comun[0][0]

    def predict(self, X):
        """
            Predicción utilizando el algoritmo KNN.
            
            Parámetros:
            -----------
            - X: Variables dependientes.
        """
        predicciones = []
        for x in X:
            predicciones.append(self.kDistance(x))

        res = np.array(predicciones)

        return res