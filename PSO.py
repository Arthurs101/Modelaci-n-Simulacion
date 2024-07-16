import numpy as np
import matplotlib.pyplot as plt

# Variables
n = 40  # número de partículas
max_iter = 100  # número máximo de iteraciones
threshold = 1e-6  # umbral de convergencia

# Generar 40 posiciones aleatorias dentro del rango -10 y 10
np.random.seed(0)  # Semilla para reproducibilidad
x = np.random.uniform(-10, 10, n)
y = np.random.uniform(-10, 10, n)

# Inicializar las velocidades de las partículas 40 velocidades aleatorias dentro de -1 y 1
vx = np.random.uniform(-1, 1, n)
vy = np.random.uniform(-1, 1, n)

# Definir la función objetivo
def funcion_objetiva(x, y):
    return (x - 3)**2 + (y - 2)**2

valores_objetivos = funcion_objetiva(x, y)

# Establecer las mejores posiciones personales
mejoresP_x = np.copy(x)
mejoresP_y = np.copy(y)
mejoresP_valores = np.copy(valores_objetivos)

# Encontrar la mejor posición global inicial
mejorG_indice = np.argmin(valores_objetivos)
mejorG_x = x[mejorG_indice]
mejorG_y = y[mejorG_indice]
mejorG_valor = valores_objetivos[mejorG_indice]

# Parámetros del PSO
# w = 0.5
# c1 = 1.5
# c2 = 1.5

# variante 1
w = 2
c1 = 2
c2 = 3

#variente 2
# w = 3
# c1 = 3
# c2 = 5

#variante 3
# w = 0.01
# c1 = 5
# c2 = 4

# Almacenar trayectorias de partículas
trayectorias_x = [np.copy(x)]
trayectorias_y = [np.copy(y)]

# Iterar el proceso de optimización
for iter in range(max_iter):
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    
    # Actualizar velocidades
    vx = w * vx + c1 * r1 * (mejoresP_x - x) + c2 * r2 * (mejorG_x - x)
    vy = w * vy + c1 * r1 * (mejoresP_y - y) + c2 * r2 * (mejorG_y - y)

    # Actualizar posiciones
    x = x + vx
    y = y + vy

    # Guardar trayectorias
    trayectorias_x.append(np.copy(x))
    trayectorias_y.append(np.copy(y))

    # Calcular el nuevo valor de la función objetivo
    valores_objetivos = funcion_objetiva(x, y)

    # Actualizar la mejor posición personal
    mejoresP_mask = valores_objetivos < mejoresP_valores
    mejoresP_x[mejoresP_mask] = x[mejoresP_mask]
    mejoresP_y[mejoresP_mask] = y[mejoresP_mask]
    mejoresP_valores[mejoresP_mask] = valores_objetivos[mejoresP_mask]

    # Actualizar la mejor posición global
    mejorG_indice = np.argmin(valores_objetivos)
    if valores_objetivos[mejorG_indice] < mejorG_valor:
        mejorG_x = x[mejorG_indice]
        mejorG_y = y[mejorG_indice]
        cambio = mejorG_valor - valores_objetivos[mejorG_indice]
        mejorG_valor = valores_objetivos[mejorG_indice]

    # Condición de parada
    if cambio < threshold:
        break

# Imprimir resultados finales
print("Mejor posición global final: (", mejorG_x, ",", mejorG_y, ")")
print("Valor de la función objetivo en la mejor posición global final:", mejorG_valor)
print("Número de iteraciones:", iter + 1)

# Crear el contour plot de la función objetivo
X = np.linspace(-10, 10, 400)
Y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(X, Y)
Z = funcion_objetiva(X, Y)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Inicio
axs[0].contourf(X, Y, Z, levels=50, cmap='viridis')
axs[0].plot(trayectorias_x[0], trayectorias_y[0], 'ro')
axs[0].plot(3, 2, 'r*', markersize=15)  # Punto mínimo teórico
axs[0].set_title('Inicio')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# Punto medio (aproximadamente)
punto_medio = len(trayectorias_x) // 2
axs[1].contourf(X, Y, Z, levels=50, cmap='viridis')
axs[1].plot(trayectorias_x[punto_medio], trayectorias_y[punto_medio], 'ro')
axs[1].plot(3, 2, 'r*', markersize=15)  # Punto mínimo teórico
axs[1].set_title('Punto Medio')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

# Final
axs[2].contourf(X, Y, Z, levels=50, cmap='viridis')
axs[2].plot(trayectorias_x[-1], trayectorias_y[-1], 'ro')
axs[2].plot(3, 2, 'r*', markersize=15)  # Punto mínimo teórico
axs[2].set_title('Final')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')

plt.tight_layout()
plt.show()
