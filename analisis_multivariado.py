# Análisis Multivariado - Miembros de Gimnasio
# Este programa analiza los datos de 970 personas de un
# gimnasio para encontrar patrones y relaciones entre
# variables como peso, altura, edad, IMC, etc.

# Importamos las librerías que vamos a usar
import pandas as pd          # Para manejar tablas de datos
import matplotlib.pyplot as plt  # Para hacer gráficos
import seaborn as sns        # Para el mapa de calor
from sklearn.preprocessing import StandardScaler  # Para estandarizar datos
from sklearn.decomposition import PCA  # Para reducir variables
import numpy as np           # Para cálculos matemáticos

# Leemos el archivo de Excel con los datos
df = pd.read_excel("Miembros_gimnasio.xlsx")

# PUNTO 1: ANÁLISIS DESCRIPTIVO
# Aquí vemos un resumen general de los datos: promedios,
# valores mínimos, máximos, etc.

# Mostramos las estadísticas básicas de cada variable numérica
# (media, desviación estándar, mínimo, máximo, etc.)
print("ESTADÍSTICAS DESCRIPTIVAS")
print(df.describe().round(2))

# Creamos histogramas para ver cómo se distribuyen los datos
# Cada gráfico muestra qué tan frecuentes son ciertos valores
df.hist(bins=15, figsize=(14, 8))
plt.tight_layout()
plt.show()

# Conclusiones del Punto 1:
print("\nCONCLUSIONES PUNTO 1")
print("- El IMC promedio es 24.9, cerca del límite entre normal y sobrepeso.")
print("- Los datos están bien distribuidos y no se observan valores atípicos.")
print("- Hay variedad en los miembros: distintas edades, pesos y rutinas.\n")

# PUNTO 2: MATRIZ DE CORRELACIÓN
# La correlación nos dice qué tanto se relacionan dos variables.
# Un valor cercano a 1 = relación fuerte positiva (suben juntas)
# Un valor cercano a -1 = relación fuerte negativa (una sube, otra baja)
# Un valor cercano a 0 = no hay relación

# Calculamos la correlación entre todas las variables numéricas
cor = df.select_dtypes('number').corr()
print("MATRIZ DE CORRELACIÓN")
print(cor.round(2))

# Hacemos un mapa de calor para visualizar las correlaciones
# Los colores rojos = correlación positiva, azules = negativa
sns.heatmap(cor, annot=True, fmt=".2f", cmap='RdBu_r', center=0)
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()

# Conclusiones del Punto 2:
print("\nCONCLUSIONES PUNTO 2")
print("- La relación más fuerte es entre Peso e IMC (0.85), lo cual es esperado.")
print("- En general las variables tienen poca relación entre sí,")
print("  lo que indica que cada una aporta información diferente.\n")

# PUNTO 3: ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
# El PCA sirve para reducir la cantidad de variables.
# En vez de usar 11 variables, intentamos resumirlas en solo 4
# "componentes" que capturen la mayor información posible.

# Seleccionamos las variables numéricas (sin el IMC, que es lo que queremos predecir)
cols = df.select_dtypes('number').columns.drop('IMC')

# Estandarizamos los datos (ponemos todo en la misma escala)
# Esto es necesario porque las variables tienen unidades muy diferentes
X = StandardScaler().fit_transform(df[cols])

# Aplicamos PCA para reducir a 4 componentes
pca = PCA(n_components=4).fit(X)

# Mostramos cuánta información captura cada componente
print("VARIANZA EXPLICADA POR CADA COMPONENTE")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f"  Componente {i+1}: {v*100:.2f}%")
total = sum(pca.explained_variance_ratio_) * 100
print(f"  TOTAL con 4 componentes: {total:.2f}%")

# Mostramos qué variables son más importantes en cada componente
print("\nPESO DE CADA VARIABLE EN LAS COMPONENTES")
print(pd.DataFrame(pca.components_.T, index=cols,
                    columns=['CP1','CP2','CP3','CP4']).round(3))

# Gráfico: cuánta información captura cada componente
acum = np.cumsum(PCA().fit(X).explained_variance_ratio_) * 100
plt.bar(range(1, len(acum)+1), PCA().fit(X).explained_variance_ratio_*100)
plt.plot(range(1, len(acum)+1), acum, 'ro-')
plt.axhline(80, color='green', linestyle='--', label='80% (ideal)')
plt.axvline(4, color='orange', linestyle='--', label='4 componentes')
plt.xlabel('Componente')
plt.ylabel('Varianza explicada (%)')
plt.title('Varianza explicada por cada componente')
plt.legend()
plt.show()

# Conclusiones del Punto 3:
print("\nCONCLUSIONES PUNTO 3")
print(f"- Con 4 componentes se captura el {total:.2f}% de la información (lo ideal es >70%).")
print("- Reducir a 4 variables NO es la mejor decisión para estos datos, porque")
print("  las variables tienen poca correlación y cada una aporta algo distinto.")
