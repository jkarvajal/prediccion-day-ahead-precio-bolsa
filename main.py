"""
Punto de entrada del proyecto de prediccion day-ahead del precio de bolsa de
energia. Documenta el orden del flujo reproducible y permite entrenar y persistir
el modelo final desde la linea de comandos.

El procesamiento, el analisis y el modelado se desarrollan en los notebooks
numerados de la carpeta notebooks/, que deben ejecutarse en orden:

    01_procesamiento_datos          Consolidacion y limpieza de las fuentes
    02_eda                          Analisis exploratorio y caracterizacion
    03_ingenieria_variables         Construccion de variables predictoras
    04_modelos_baseline             Lineas base de persistencia
    05_modelos_mensuales            Modelos mensuales (analisis de soporte)
    06_modelos_diarios              Modelos diarios (analisis de soporte)
    07_modelos_horarios_dayahead    Modelos horarios day-ahead (nucleo)
    08_recalibracion_y_significancia Recalibracion y prueba de significancia

Uso:
    python main.py            Muestra el orden del flujo.
    python main.py --train    Entrena y persiste el modelo final.
"""

import sys


PIPELINE = [
    ("01_procesamiento_datos", "Consolidacion y limpieza de las fuentes"),
    ("02_eda", "Analisis exploratorio y caracterizacion temporal"),
    ("03_ingenieria_variables", "Construccion de variables predictoras"),
    ("04_modelos_baseline", "Lineas base de persistencia"),
    ("05_modelos_mensuales", "Modelos mensuales (analisis de soporte)"),
    ("06_modelos_diarios", "Modelos diarios (analisis de soporte)"),
    ("07_modelos_horarios_dayahead", "Modelos horarios day-ahead (nucleo)"),
    ("08_recalibracion_y_significancia", "Recalibracion y significancia"),
]


def mostrar_flujo():
    print("Flujo reproducible del proyecto (ejecutar los notebooks en orden):\n")
    for nombre, desc in PIPELINE:
        print(f"  {nombre:34s} {desc}")
    print("\nPara entrenar y persistir el modelo final: python main.py --train")


def main():
    if "--train" in sys.argv:
        from src.train_final_model import train_final_model
        train_final_model()
    else:
        mostrar_flujo()


if __name__ == "__main__":
    main()
