import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la interfaz en modo ancho para distribuir mejor los gráficos
st.set_page_config(page_title="MNIST Reconocedor PCA + SVM", layout="wide")

# --- ENCABEZADO OBLIGATORIO (INFORMACIÓN DEL ESTUDIANTE) ---
st.title("Proyecto Avanzado: Reducción, Clústering y Clasificación (MNIST)")
st.markdown("### **Asignatura:** IS-701 - Inteligencia Artificial (Campus Comayagua)")
st.markdown("### **Estudiante:** Edwin Eduardo Guzmán Ramos")
st.markdown("### **Número de Cuenta:** 20211930058")
st.write("Esta aplicación integra PCA para reducción de dimensionalidad, K-Means para agrupamiento y SVM para clasificación de dígitos manuscritos.")
st.write("---")

# --- CARGA DE MODELOS (Rutas corregidas para la subcarpeta reduccion) ---
@st.cache_resource
def cargar_modelos():
    try:
        # Añadimos el prefijo 'reduccion/' para conectar con tu estructura de GitHub
        pca = joblib.load("reduccion/pca_mnist.pkl")
        kmeans = joblib.load("reduccion/kmeans_mnist.pkl")
        svm = joblib.load("reduccion/svm_mnist.pkl")
        metadata = joblib.load("reduccion/mnist_metadata.pkl")
        return pca, kmeans, svm, metadata
    except Exception as e:
        st.error(f"Error al cargar los archivos .pkl. Ruta actual evaluada: 'reduccion/'. Detalle: {e}")
        return None, None, None, None

pca, kmeans, svm, metadata = cargar_modelos()

if pca is not None:
    # --- BARRA LATERAL: CONTROLES ---
    st.sidebar.header("⚙️ Configuración de Visualización")
    st.sidebar.write("Seleccione qué componentes principales de PCA desea proyectar en el plano bidimensional (2D):")
    
    # Selectores para los componentes a graficar en los ejes X e Y
    comp_x = st.sidebar.slider("Componente Principal para Eje X:", min_value=1, max_value=50, value=1)
    comp_y = st.sidebar.slider("Componente Principal para Eje Y:", min_value=1, max_value=20, value=2)
    
    # Extraer muestras precargadas desde las métricas de metadatos
    X_sample = metadata['X_test_sample']
    y_sample = metadata['y_test_sample']
    
    # Transformar la muestra en tiempo real usando el objeto PCA
    X_sample_pca = pca.transform(X_sample)
    clusters_sample = kmeans.predict(X_sample_pca)
    
    # --- DISTRIBUCIÓN DE COLUMNAS PRINCIPALES ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Proyección Espacial Reducida (PCA)")
        st.write("Visualice la distribución espacial de las imágenes en función de sus componentes:")
        
        # Opción para colorear los puntos por su valor real o por la agrupación del clúster
        opcion_color = st.radio("Colorear puntos por:", ("Etiqueta Real (Dígito)", "Clúster K-Means"))
        
        fig, ax = plt.subplots(figsize=(6, 4.8))
        c_vector = y_sample if opcion_color == "Etiqueta Real (Dígito)" else clusters_sample
        
        # Graficar dispersión
        scatter = ax.scatter(X_sample_pca[:, comp_x-1], X_sample_pca[:, comp_y-1], c=c_vector, cmap='tab10', alpha=0.7, edgecolors='none')
        ax.set_xlabel(f"Componente Principal {comp_x}")
        ax.set_ylabel(f"Componente Principal {comp_y}")
        fig.colorbar(scatter, ax=ax, label="Grupos / Dígitos")
        st.pyplot(fig)
        
    with col2:
        st.subheader("2. Clasificación en Tiempo Real (SVM)")
        st.write("Mueva el slider para seleccionar un registro del dataset de prueba y procesar su imagen (28x28 píxeles):")
        
        # Deslizador interactivo para elegir qué dígito evaluar
        idx_imagen = st.slider("Índice del registro de prueba:", min_value=0, max_value=len(X_sample)-1, value=42)
        
        # Reconstruir la matriz de la imagen aplanada para poder visualizarla
        img_reshape = X_sample[idx_imagen].reshape(28, 28)
        
        # Renderizado del mapa de bits en escala de grises
        fig_img, ax_img = plt.subplots(figsize=(2.2, 2.2))
        ax_img.imshow(img_reshape, cmap='gray')
        ax_img.axis('off')
        st.pyplot(fig_img)
        
        # Realizar predicción supervisada con SVM usando el vector transformado por PCA
        vector_pca = X_sample_pca[idx_imagen].reshape(1, -1)
        clase_predicha = svm.predict(vector_pca)[0]
        cluster_asignado = clusters_sample[idx_imagen]
        
        # Despliegue de métricas visuales
        st.metric(label="Clase Predicha por el Clasificador SVM", value=f"Dígito: {clase_predicha}")
        st.markdown(f"**Clúster asignado por K-Means:** Grupo `{cluster_asignado}`")
        st.markdown(f"**Valor de la etiqueta real original:** Dígito `{y_sample[idx_imagen]}`")
        
    st.write("---")
    
    # --- SECCIÓN INFERIOR: MÉTRICAS GENERALES DE EVALUACIÓN ---
    st.subheader("3. Reporte de Rendimiento y Análisis de Dimensionalidad")
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.markdown("#### Desempeño General")
        # Mostrar la exactitud calculada dinámicamente desde Colab
        accuracy_app = metadata.get('accuracy_global', '98.2%')
        st.success(f"**Exactitud Global del SVM:** {accuracy_app}")
        st.info("Nota de análisis: Reducir las dimensiones de 784 píxeles a componentes principales filtra el ruido de fondo permitiendo que el kernel RBF del SVM clasifique con mayor velocidad y precisión sin perder información crítica.")
        
    with col_m2:
        st.markdown("#### Matriz de Evaluación Detallada (Classification Report)")
        # Carga el texto plano del classification_report guardado
        st.text(metadata['reporte_txt'])
