import numpy as np
import math

def SGD(self, X, y, batch_size, learning_rate, eps, max_steps):
    indexes = list(range(len(X)))
    for s in range(max_steps):
        # Случайным образом выбираем часть примеров из всех имеющихся
        sample_ind = np.random.choice(indexes, batch_size, replace = False)
        X_example = X[sample_ind]
        y_answer = y[sample_ind]
            
            #Рассчитывает градиент
        res = self.update_mini_batch(X_example, y_answer, learning_rate, eps) 
        if res:  # спуск сошёлся
            return 1 
    return 0 #спуск не достиг минимума за отведённое время

def update_mini_batch(self, X, y, learning_rate, eps):
    nabla_j = compute_grad_analytically(self, X, y)
    j_old = J_quadratic(self, X, y)
    delta_w = -1 * learning_rate * nabla_j
    self.w += delta_w
    j_new = J_quadratic(self, X, y)
    
    if math.fabs(j_old - j_new) < eps:
        return 1
    else:
        return 0