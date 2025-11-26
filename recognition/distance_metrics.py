"""
Módulo con métricas de distancia para reconocimiento de comandos.
Todos utilizan std_energy (desviación estándar) para mejor identificación.

Autor: Lab Señales
Fecha: 2024
"""

import numpy as np


def euclidean(energies, model_energies, model_std_energies=None):
    """
    Distancia euclidiana clásica (línea base).
    
    Fórmula:
        d = sqrt(sum_i(E_input_i - E_modelo_i)^2)
    
    Parámetros:
    -----------
    energies : array
        Vector de energías de entrada: [E_1, E_2, ..., E_N]
    model_energies : array
        Vector de energías del modelo (media)
    model_std_energies : array (ignorado)
        No se usa en esta métrica
    
    Retorna:
    --------
    distance : float
        Distancia euclidiana
    """
    differences = energies - model_energies
    distance = np.sqrt(np.sum(differences ** 2))
    return distance


def weighted_euclidean(energies, model_energies, model_std_energies, eps=1e-8):
    """
    Distancia euclidiana ponderada por variabilidad.
    Las bandas con menor variabilidad (std pequeño) contribuyen más.
    
    Fórmula:
        w_i = 1 / (std_i + eps)
        d = sqrt(sum_i(w_i * (E_input_i - E_modelo_i)^2))
    
    Intuitición: bandas estables (std pequeño) son más discriminativas,
                 bandas ruidosas (std grande) contribuyen menos.
    
    Parámetros:
    -----------
    energies : array
        Vector de energías de entrada
    model_energies : array
        Vector de energías del modelo (media)
    model_std_energies : array
        Vector de desviaciones estándar por banda
    eps : float
        Pequeño valor para evitar división por cero
    
    Retorna:
    --------
    distance : float
        Distancia euclidiana ponderada
    """
    # Calcular pesos: bandas estables tienen pesos mayores
    weights = 1.0 / (model_std_energies + eps)
    
    # Normalizar pesos para que el rango sea comparable
    weights = weights / np.mean(weights)
    
    # Calcular diferencias ponderadas
    differences = energies - model_energies
    weighted_sq_diff = weights * (differences ** 2)
    
    # Distancia ponderada
    distance = np.sqrt(np.sum(weighted_sq_diff))
    return distance


def nll_gaussian(energies, model_energies, model_std_energies, eps=1e-8):
    """
    Negative Log-Likelihood bajo distribución gaussiana.
    Mide qué tan probable es que la entrada pertenezca a este modelo.
    
    Fórmula (negativa para que menor = mejor):
        NLL = 0.5 * sum_i(log(2π*var_i) + (E_input_i - E_modelo_i)^2 / var_i)
    
    Interpretación: combina componentes de precisión (std) y error de media.
                   Un modelo es "mejor" si tiene NLL más bajo.
    
    Parámetros:
    -----------
    energies : array
        Vector de energías de entrada
    model_energies : array
        Vector de energías del modelo (media)
    model_std_energies : array
        Vector de desviaciones estándar por banda
    eps : float
        Pequeño valor para evitar división por cero
    
    Retorna:
    --------
    nll : float
        Negative Log-Likelihood (menor es mejor)
    """
    # Calcular varianzas
    variances = (model_std_energies + eps) ** 2
    
    # Calcular NLL: log-verosimilitud negativa
    differences = energies - model_energies
    
    # Componente 1: log de la desviación estándar (precision term)
    log_2pi_var = np.log(2 * np.pi * variances)
    
    # Componente 2: error normalizado por varianza
    normalized_sq_diff = (differences ** 2) / variances
    
    # NLL total
    nll = 0.5 * np.sum(log_2pi_var + normalized_sq_diff)
    return nll


# Diccionario de métricas disponibles
AVAILABLE_METRICS = {
    'euclidean': euclidean,
    'weighted_euclidean': weighted_euclidean,
    'nll_gaussian': nll_gaussian,
}

# Nombres descriptivos para UI
METRIC_NAMES = {
    'euclidean': 'Euclidiana (Línea Base)',
    'weighted_euclidean': 'Euclidiana Ponderada',
    'nll_gaussian': 'Verosimilitud Gaussiana',
}

# Descripción corta para tooltips
METRIC_DESCRIPTIONS = {
    'euclidean': 'Distancia clásica sin usar std_energy',
    'weighted_euclidean': 'Bandas estables contribuyen más',
    'nll_gaussian': 'Probabilidad gaussiana (mejor si menor)',
}


def calculate_distance(energies, model_energies, model_std_energies, 
                      distance_method='euclidean'):
    """
    Calcula la distancia usando el método especificado.
    
    Parámetros:
    -----------
    energies : array
        Vector de energías de entrada
    model_energies : array
        Vector de energías del modelo (media)
    model_std_energies : array
        Vector de desviaciones estándar por banda
    distance_method : str
        Nombre del método ('euclidean', 'weighted_euclidean', 'nll_gaussian')
    
    Retorna:
    --------
    distance : float
        Distancia calculada
    
    Raises:
    -------
    ValueError
        Si el método especificado no existe
    """
    if distance_method not in AVAILABLE_METRICS:
        raise ValueError(
            f"Método de distancia desconocido: {distance_method}. "
            f"Opciones: {list(AVAILABLE_METRICS.keys())}"
        )
    
    metric_func = AVAILABLE_METRICS[distance_method]
    return metric_func(energies, model_energies, model_std_energies)
