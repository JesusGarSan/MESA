import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configurar rutas y cargar datos
csv_path = "external/examples/GC-FID/optimization.csv"  
output_dir = "external/examples/GC-FID/heatmaps"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Carpeta creada: {output_dir}")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {csv_path}")
    exit()


df.index = [round(float(x), 2) if isinstance(x, (int, float)) or str(x).replace('.','',1).isdigit() else x for x in df.index]
df.columns = [round(float(x), 2) if isinstance(x, (int, float)) or str(x).replace('.','',1).isdigit() else x for x in df.columns]

df = df.round(2)

# 2. Identificar las columnas de Suma de Cuadrados (SS) dinámicamente
ss_columns = [col for col in df.columns if col.startswith('SS_')]

# 3. Identificar los tipos de ventana únicos presentes en los datos
unique_windows = df['window'].unique()

print(f"Detectadas {len(ss_columns)} métricas SS: {ss_columns}")
print(f"Detectados {len(unique_windows)} tipos de ventana: {unique_windows}")
print("Generando mapas de calor con redondeo estricto...")

# 4. Iterar por cada tipo de ventana y por cada métrica SS
for window_type in unique_windows:
    df_window = df[df['window'] == window_type]
    
    for ss_metric in ss_columns:
        # Creamos la tabla pivote
        pivot_df = df_window.pivot_table(
            values=ss_metric, 
            index='window_length_fraction', 
            columns='hop_fraction'
        )
        
        # Invertimos el eje Y para que las fracciones más grandes queden arriba
        pivot_df = pivot_df.iloc[::-1]
        
        # --- SOLUCIÓN DE REDONDEO ---
        # 1. Redondeamos los valores internos de la matriz SS a 3 decimales
        pivot_df = pivot_df.round(3)
        

        # Configurar el gráfico de matplotlib
        plt.figure(figsize=(16, 8))
        
        # Generar el mapa de calor con Seaborn
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt=".3f", # Asegura que los números internos muestren ceros a la derecha si es necesario (ej: 0.950)
            cmap="viridis", 
            cbar_kws={'label': 'Valor SS'}
        )
        
        # Añadir títulos y etiquetas de los ejes
        clean_name = ss_metric.replace('SS_', '').capitalize()
        plt.title(f"Mapeo de {clean_name}\nVentana: {window_type}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Hop Fraction", fontsize=12)
        plt.ylabel("Window Length Fraction", fontsize=12)
        
        plt.tight_layout()
        
        # 5. Guardar la imagen con un nombre descriptivo
        filename = f"{output_dir}/heatmap_{window_type}_{ss_metric}.png"
        plt.savefig(filename, dpi=300)
        plt.close() 
        
        print(f" -> Guardado: {filename}")

print("\n¡Proceso completado con éxito!")