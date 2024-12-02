import numpy as np
from utils import moody_friction_factor


def duns_ros_dpdl(P: float, angle: float, d: float, e: float, rho_g: float, rho_l: float, mu_g: float, mu_l: float, sigma_l, vsg: float, vsl: float): 
    angle = 0.01745329 * angle
    flowpat = FlowPattern(d, vsg, vsl, rho_l, sigma_l)
    dpdl = 0.0
    if flowpat == "BUBBLE":
        dpdl = BubbleDpdl(angle, d, vsg, vsl, rho_g, rho_l, mu_l, mu_g, sigma_l, P, e)
    
    elif flowpat == "SLUG":
        dpdl = SlugDpdl(angle, d, vsg, vsl, rho_g, rho_l, mu_l, mu_g, sigma_l, P, e)
    
    elif flowpat == "MIST":
        dpdl = MistDpdl(angle, d, vsg, vsl, rho_g, rho_l, mu_g, P, e)
    
    elif flowpat == "TRANSITION":
        dpdl = TransitionDpdl(angle, d, vsg, vsl, rho_g, rho_l, mu_l, mu_g, sigma_l, P, e)    
    
    return dpdl[0]


def FlowPattern(d, vsg, vsl, rhoL, sigma_l):
    flowPat = ""
    NLV = 1.938 * vsl * np.power(rhoL / sigma_l, 0.25)
    NGV = 1.938 * vsg * np.power(rhoL / sigma_l, 0.25)
    ND = 120.872 * d * np.sqrt(rhoL/ sigma_l)
    # NL = 0.15726 * muL * np.power(1 / (rhoL * np.power(sigma_l, 3)), 0.25);   

    L1 = 0.0
    L2 = 0.0
    
    if ND <= 19.9526:
        L1 = 2.0
    
    elif ND > 19.9526 and ND < 31.6228:
        L1 = 3.7097 - 0.0857 * ND
    
    elif ND >= 31.6228:
        L1 = 1.0

    if ND <= 63.0957:
        L2 = 0.01165 * ND + 0.14671
    
    else:
        L2 = 1.1

    # Flow pattern transition
    NGVBS = L1 + L2 * NLV
    NGVSTR = 50 + 36 * NLV
    NGVTRM = 75 + 84 * np.power(NLV, 0.75)

    if NGV < NGVBS:
        flowPat = "BUBBLE"
    
    elif (NGVBS < NGV) and (NGV < NGVSTR):
        flowPat = "SLUG"
    
    elif NGV > NGVTRM:
        flowPat = "MIST"
    
    elif NGV > NGVSTR and NGV < NGVTRM:
        flowPat = "TRANSITION"
    

    return flowPat


def GetF1(NL: float) -> float:
    nl_values = [0.002, 0.014, 0.02, 0.03, 0.04, 0.06, 0.09, 0.1, 
                0.14, 0.2, 0.24495, 0.3, 0.4, 0.6, 0.9, 1.414, 2.0 ]
    F1_values = [1.2, 1.2, 1.291, 1.291, 1.5, 1.7, 1.9, 1.951, 2.041,
                1.999, 1.951, 1.851, 1.70029, 1.451, 1.151, 0.89968, 0.79953]
    for i in range(len(nl_values)):
        nl_values[i] = np.log10(nl_values[i])
        F1_values[i] = np.log10(F1_values[i])
    
    LogN = max(nl_values[0], min(nl_values[len(nl_values) - 1], np.log10(NL)))
    F1 = np.power(10, np.interp(LogN, nl_values, F1_values))
    
    return F1


def GetF2(NL):
    nl_values = [0.002, 0.005, 0.01, 0.01414, 0.02, 0.04, 0.08, 0.09, 0.1,
                0.1189, 0.1414, 0.16817, 0.2, 0.3, 0.7, 1.0, 2.0 ]
    F2_values = [0.24, 0.24, 0.24, 0.24, 0.269, 0.48, 0.84853, 0.9, 0.94868, 
                1.0488, 1.0488, 1.0488, 1.0488, 1.0, 0.84853, 0.8, 0.7]
    for i in range(len(nl_values)):
        nl_values[i] = np.log10(nl_values[i])
        F2_values[i] = np.log10(F2_values[i])
    
    LogN = max(nl_values[0], min(nl_values[len(nl_values) - 1], np.log10(NL)))
    F2 = np.power(10, np.interp(LogN, nl_values, F2_values))
    return F2


def GetF3(NL):
    nl_values = [0.002, 0.003, 0.0034641, 0.004, 0.005, 0.005477, 0.006, 0.007, 0.009,
                0.01, 0.02, 0.04, 0.07, 0.1, 0.14142, 0.2, 0.5, 1.0, 2.0 ]
    F3_values = [0.74891, 0.74891, 0.74891, 0.77353, 0.77353, 0.77371, 0.825, 0.925,
                1.125, 1.173, 1.824, 2.525, 2.925, 3.125, 3.325, 3.525, 3.725, 3.925, 4.125]
    for i in range(len(nl_values)):
        nl_values[i] = np.log10(nl_values[i])
        F3_values[i] = np.log10(F3_values[i])
    
    LogN = max(nl_values[0], min(nl_values[len(nl_values) - 1], np.log10(NL)))
    F3 = np.power(10, np.interp(LogN, nl_values, F3_values))
    
    return F3


def GetF4(NL):
    nl_values = [ 0.002, 0.003, 0.005, 0.009, 0.03, 0.04, 0.05, 
                0.06, 0.07, 0.08, 0.1, 0.3, 0.6, 1.0, 2.0]
    F4_values = [-18, -5, 8, 21, 44, 49.5, 52, 56, 56, 56.5, 57, 57, 57, 57, 57 ]
    for i in range(len(nl_values)):
        nl_values[i] = np.log10(nl_values[i])
    
    LogN = max(nl_values[0], min(nl_values[len(nl_values) - 1], np.log10(NL)))
    F4 = np.interp(LogN, nl_values, F4_values)
    
    return F4


def GetF5(NL):
    nl_values = [ 0.002, 0.003, 0.006, 0.02, 0.024494, 0.03, 0.04, 0.04472, 0.05, 0.054772, 0.06,
                0.07, 0.09, 0.1, 0.11892, 0.14142, 0.168178, 0.2, 0.4, 0.7, 1.0, 1.141421, 2.0]
    F5_values = [ 0.22, 0.20976, 0.2, 0.17, 0.16492, 0.16, 0.14491, 0.1349, 0.13, 0.12, 0.10954, 
                0.09487, 0.007, 0.006, 0.048, 0.042, 0.042, 0.046, 0.06, 0.08, 0.09487, 0.11892, 0.14142]
    for i in range(len(nl_values)):
        nl_values[i] = np.log10(nl_values[i])
        F5_values[i] = np.log10(F5_values[i])
    
    LogN = max(nl_values[0], min(nl_values[len(nl_values) - 1], np.log10(NL)))
    F5 = np.power(10, np.interp(LogN, nl_values, F5_values))
    
    return F5


def GetF6(NL):
    nl_valus = [ 0.002, 0.003, 0.004, 0.005, 0.007, 0.009, 0.01, 0.011892, 
                0.014142, 0.016818, 0.02, 0.024495, 0.03, 0.05, 0.07, 0.09, 
                0.1, 0.11892, 0.14142, 0.16818, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0 ]
    F6_values = [0.86, 0.50, 0.30, 0.18, 0.04, -0.04, -0.08, -0.12, -0.14, -0.12, 
                -0.08, 0.08, 0.36, 0.96, 1.46, 1.98, 2.10, 2.18, 2.10, 2.04, 1.98,
                1.86, 1.80, 1.76, 1.765, 1.765, 1.765 ]
    for i in range(len(nl_valus)):
        nl_valus[i] = np.log10(nl_valus[i])
    
    LogN = max(nl_valus[0], min(nl_valus[len(nl_valus) - 1], np.log10(NL)))
    F6 = np.interp(LogN, nl_valus, F6_values)
    
    return F6


def GetF7(NL):
    nl_values = [ 0.002, 0.003, 0.005, 0.02, 0.07, 0.1, 0.14142, 0.2,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.4142, 2.0 ]
    F7_values = [0.13, 0.12490, 0.10954, 0.07, 0.04499, 0.04, 0.035, 0.032, 0.03,
                0.028, 0.02698, 0.026, 0.026, 0.02498, 0.02498, 0.02449, 0.024, 0.024]
    for i in range(len(nl_values)):
        nl_values[i] = np.log10(nl_values[i])
        F7_values[i] = np.log10(F7_values[i])
    
    LogN = max(nl_values[0], min(nl_values[len(nl_values) - 1], np.log10(NL)))
    F7 = np.power(10, np.interp(LogN, nl_values, F7_values))
    
    return F7


def Getf2(f1, superficialgasvelocity, ND, superficialliquidvelocity):
    f2 = 0
    a = f1 * superficialgasvelocity * np.power(ND, 0.666667) / superficialliquidvelocity
    x = [ 0.3162, 31.6228 ]
    y = [ 1, 0.22 ]
    if a <= 0.3162:
        f2 = 1
    
    elif a > 0.3162 and a <= 31.6228:
        f2 = np.interp(a, x, y)

    elif a > 31.6228:
        f2 = 0.20
    
    return f2


def LiquidHoldUp(d, vsg, vsl, rhoL, muL, sigmaL):
    F3prime = 0.0
    F6prime = 0.0
    S = 0.0
    vm = vsl + vsg

    NLV = 1.938 * vsl * np.power(rhoL / sigmaL, 0.25)
    NGV = 1.938 * vsg * np.power(rhoL / sigmaL, 0.25)
    ND = 120.872 * d * np.sqrt(rhoL / sigmaL)
    NL = 0.15726 * muL * np.power(1 / (rhoL * np.power(sigmaL, 3)), 0.25) 

    flowPat = FlowPattern(d, vsg, vsl, rhoL, sigmaL)
    HL = 0.0
    f1 = GetF1(NL)
    f2 = GetF2(NL)
    f3 = GetF3(NL)
    f4 = GetF4(NL)
    f5 = GetF5(NL)
    f6 = GetF6(NL)
    f7 = GetF7(NL)
    if flowPat == "BUBBLE":
        F3prime = f3 - (f4 / ND)
        S = f1 + (NLV * f2) + (F3prime * np.power((NGV / (1 + NLV)), 2))
    
    elif flowPat == "SLUG":
        F6prime = 0.029 * ND + f6
        S = (1 + f5) * (np.power(NGV, 0.982) + F6prime) / np.power(1 + f7 * NLV, 2)
    
    elif flowPat == "MIST":
        S = 0.0

    vs = S /(1.98 *  np.power(rhoL / sigmaL, 0.25))
    if flowPat == "BUBBLE" or flowPat == "SLUG":
        HL = (vs - vm + np.sqrt(np.power(vm - vs, 2) + 4 * vs * vsl)) / (2 * vs)
    
    elif flowPat == "TRANSITION" or flowPat == "MIST":
        HL = 0
    
    return HL


def BubbleDpdl(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL, P, e):
    vm = vsl + vsg
    lambdaL = vsl / vm
    g = 32.174
    ND = d * np.sqrt(rhoL * g / sigmaL)

    HL = LiquidHoldUp(d, vsg, vsl, rhoL, muL, sigmaL)
    muNS = muL * lambdaL + muG * (1 - lambdaL)
    rhoNS = rhoL * lambdaL + rhoG * (1 - lambdaL)
    rhoS = rhoL * HL + rhoG * (1 - HL)

    Re = 1488 * rhoL * vsl * d / muL
    f1 = moody_friction_factor(e/d, Re)
    f2 = Getf2(f1, vsg, ND, vsl)
    f3 = 1 + (f1 / 4) * np.sqrt(vsg / (50 * vsl))
    frictionfactor = f1 * f2 / f3
    dpdzf = frictionfactor * vm * vsl * rhoL / (2 * d * g)
    dpdzel = rhoS * np.sin(angle)
    EK = vm * vsg * rhoNS / (P * 144)

    dpdztm = 0
    dpdztpsi = 0
    dpdztm = (dpdzel + dpdzf)
    dpdztpsi = dpdztm/144
    return dpdztpsi


def SlugDpdl(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL, P, e):
    g = 32.174
    vm = vsl + vsg
    ND = d * np.sqrt(rhoL * g / sigmaL)
    HL = LiquidHoldUp(d, vsg, vsl, rhoL, muL, sigmaL)
    Re = rhoL * vsl * d / muL
    f1 = moody_friction_factor(e/d, Re)
    f2 = Getf2(f1, vsg, ND, vsl)
    f3 = 1 + (f1 / 4) * np.sqrt(vsg / (50 * vsl))
    frictionfactor = f1 * f2 / f3
    dpdzf = frictionfactor * vm * vsl * rhoL / (2 * d)

    rhoS = rhoL * HL + rhoG * (1 - HL)
    dpdzel = rhoS * g*  np.sin(angle)
    # EK = vm * vsg * rhoS / (P * g * 144)
    EK = vm * vsg * rhoS / P
    dpdztm = 0 
    dpdztpsi = 0
    if EK > 0.95:
        dpdztm = (dpdzel + dpdzf)
    
    else:
        dpdztm = (dpdzel + dpdzf) / (1 - EK)
    
    dpdztpsi = dpdztm
    return dpdztpsi


def MistDpdl(angle, d, vsg, vsl, rhoG, rhoL, muG, P, e):
    g = 32.174
    vm = vsg + vsl
    lambdaL = vsl / vm
    Re = rhoG * vsg * d / muG
    frictionfactor = moody_friction_factor(e/d, Re)
    dpdzf = (frictionfactor * vsg * vsg * rhoG / (2 * d))
    noslipdensity = rhoL * lambdaL + rhoG * (1 - lambdaL)
    dpdzel = noslipdensity * np.sin(angle) * g
    EK = vm * vsg * noslipdensity / P
    dpdztm = 0
    dpdztpsi = 0
    if EK > 0.95:
        dpdztm = (dpdzel + dpdzf)
    
    else:
        dpdztm = (dpdzel + dpdzf) / (1 - EK)
    

    dpdztpsi = dpdztm
    return dpdztpsi


def TransitionDpdl(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL, P, e):
    g = 32.174
    NLV = vsl * np.power(rhoL / (g * sigmaL), 0.25)
    NGV = vsg * np.power(rhoL / (g * sigmaL), 0.25)

    NGVSTR = 50 + 36 * NLV
    NGVTRM = 75 + 84 * np.power(NLV, 0.75)
    A = (NGVTRM - NGV) / (NGVTRM - NGVSTR)
    dpdzts = SlugDpdl(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL, P, e)
    dpdztm = MistDpdl(angle, d, vsg, vsl, rhoG, rhoL, muG, P, e)
    dpdztpsi = A * dpdzts + (1 - A) * dpdztm
    return dpdztpsi





if __name__ == "__main__":

    dpdl1 = duns_ros_dpdl(1700, 90, 0.5, 0.00006, 5.88, 47.61, 0.016, 0.97, 8.41, 3.86, 3.97)
    print(f"{dpdl1 = }")


