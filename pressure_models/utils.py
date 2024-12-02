from numpy import power, sqrt, log10
from scipy.optimize import fsolve


def friction_factor(d: float, e: float, Re: float) -> float:
    fn = 0
    if e == 0:
        fn = smooth_pipe_friction_factor(Re)
    else:
        ed = e/d
        fn = moody_friction_factor(ed, Re)
    
    return fn


def moody_friction_factor(ed: float, Re: float) -> float:
    f = 0.0
    if Re < 0.0:
        raise ValueError("Reynolds number cannot be zero")
    elif Re < 2000:
        f = 64.0/Re
    
    if ed > 0.5:
        print(f"epsilon - diameter ratio {ed} is not on the Moody chart")
    
    if Re < 4000.0:
        print(f"Reynolds number {Re} is in the Transition region")
    
    def coolbrook_func(f):
        return 1.0/sqrt(f) + 2.0*log10(ed/3.7 + 2.51/(Re*sqrt(f)))

    f0 = 1/(1.8*power(log10(6.9/Re + power(ed/3.7, 1.11)), 2))
    f = fsolve(coolbrook_func, f0)

    if f < 0:
        raise ValueError(f"Friction factor cannot be negative!")
    
    return f


def smooth_pipe_friction_factor(Re: float) -> float:
    f = 0.0
    if Re < 0:
        raise ValueError(f"Reynolds number = {Re}, cannot be negative")
    elif Re < 2000:
        f = 64/Re
    else:
        f = 0.184 * power(Re, -0.2)
    
    return f



