# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not isinstance(n, int) or n <= 0:
            return None

        k = np.arange(0, n)
        x_k = np.zeros(n)
        for i in range(len(k)):
            x_k[i] = np.cos(k[i]* np.pi / (n-1))
        return x_k

    except Exception:
        return None


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not isinstance(n, int) or n <= 0:
            return None

        omega_j = np.zeros(n)
        for j in range(n):
            if j == 0 or j == n - 1:
                omega_j[j] = 0.5 * ((-1) ** j)
            else:
                omega_j[j] = (-1) ** j
        return omega_j
    

    except Exception:
        return None


def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not (isinstance(xi, np.ndarray)):
            return None
        if not (isinstance(yi, np.ndarray)):
            return None
        if not (isinstance(wi, np.ndarray)):
            return None
        if not (isinstance(x, np.ndarray)):
            return None
        if not (xi.shape == yi.shape == wi.shape):
            return None
        
        n = x.shape[0]
        m = xi.shape[0]
        P_x = np.zeros(n)
        for i in range(n):
            numerator = 0.0
            denominator = 0.0
            for j in range(m):
                if x[i] == xi[j]:
                    numerator = yi[j]
                    denominator = 1.0
                    break
                temp = wi[j] / (x[i] - xi[j])
                numerator += temp * yi[j]
                denominator += temp
            P_x[i] = numerator / denominator
        return P_x
    
    except Exception:
        return None


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not (isinstance(xr, (int, float, list, np.ndarray))):
            return None
        if not (isinstance(x, (int, float, list, np.ndarray))):
            return None
        
        xr_array = np.array(xr)
        x_array = np.array(x)
        if xr_array.shape != x_array.shape:
            return None
        return np.max(np.abs(xr_array - x_array))
    except Exception:
        return None
    
def f_1(x):
    x = np.asarray(x, dtype=float)
    sgnx = np.where(x != 0, np.abs(x) / x, 0)
    return sgnx * x + x**2
        
def f_2(x):
    x = np.asarray(x, dtype=float)  
    sgnx = np.sign(x)               
    return sgnx * x**2

def f_3(x):
    x = np.asarray(x, dtype=float)
    return np.abs(np.sin(5 * x))**3
    
def f_4a(
            x: float | int 
            ) -> float | None:
        try:
            if not isinstance(x, (int, float)):
                return None
            
            a = [ 1, 25, 100]
            y = np.zeros(len(a))
            for i in range(len(a)):
                y[i] = 1 / (1 + a[i] * x**2)
            return y    

        except Exception:
            return None
        
def f_5(x):
    x = np.asarray(x)           # zamieniamy wejście na tablicę (jeśli nie jest)
    sgnx = np.zeros_like(x)     # domyślnie 0
    sgnx[x > 0] = 1
    sgnx[x < 0] = -1
    return sgnx
