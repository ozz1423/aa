# Script en Python
## Este script devuelve el promedio de los alumnos con 15 y 16 años, y el promedio de alumnos con 17 y 18 años
### Utilizamos diccionarios para almacenar la información de cada alumno para después agruparla en listas para el cálculo del promedio

El código Python utilizado para calcular los promedios de las notas por grupos de edad es el siguiente:

```python
alumnos = {
    1001: {"Age": 17, "Score": 45},
    1002: {"Age": 18, "Score": 61},
    1003: {"Age": 15, "Score": 97},
    1004: {"Age": 17, "Score": 74},
    1005: {"Age": 17, "Score": 88},
    1006: {"Age": 18, "Score": 100},
    1007: {"Age": 15, "Score": 53},
    1008: {"Age": 15, "Score": 92},
    1009: {"Age": 17, "Score": 86},
    1010: {"Age": 16, "Score": 37},
    1011: {"Age": 17, "Score": 68},
    1012: {"Age": 17, "Score": 90},
    1013: {"Age": 17, "Score": 22},
    1014: {"Age": 17, "Score": 81},
    1015: {"Age": 18, "Score": 57},
    1016: {"Age": 15, "Score": 95},
    1017: {"Age": 18, "Score": 49},
    1018: {"Age": 18, "Score": 66},
    1019: {"Age": 18, "Score": 63},
    1020: {"Age": 17, "Score": 31},
    1021: {"Age": 16, "Score": 84},
    1022: {"Age": 15, "Score": 42},
    1023: {"Age": 16, "Score": 77},
    1024: {"Age": 18, "Score": 69},
    1025: {"Age": 18, "Score": 25},
    1026: {"Age": 16, "Score": 58},
    1027: {"Age": 16, "Score": 99},
    1028: {"Age": 16, "Score": 75},
    1029: {"Age": 18, "Score": 80},
    1030: {"Age": 18, "Score": 93},
}

agrupar15_16 = []
agrupar17_18 = []

for alumno in alumnos.values():
    if alumno["Age"] in [15, 16]:
        agrupar15_16.append(alumno["Score"])
    elif alumno["Age"] in [17, 18]:
        agrupar17_18.append(alumno["Score"])

print("Promedio alumnos 15 y 16: " + str(sum(agrupar15_16)/len(agrupar15_16)))
print("Promedio alumnos 17 y 18: " + str(sum(agrupar17_18)/len(agrupar17_18)))