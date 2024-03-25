def norm2frame(data, frame):
    import numpy as np
    from scipy.interpolate import interp1d
    x = np.array (range (0, len (data)))
    new_x = np.linspace (x.min (), x.max (), frame)
    new_y = interp1d (x, data, kind='cubic') (new_x)
    return new_y