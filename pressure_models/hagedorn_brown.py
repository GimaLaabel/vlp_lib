import numpy as np
from utils import moody_friction_factor


def hagedorn_brown_dpdl(P: float, angle: float, d: float, e: float, rho_g: float, rho_l: float, mu_g: float, mu_l: float, sigma_l, vsg: float, vsl: float):   
    angle = 0.01745329 * angle
    vm = vsg + vsl
    g = 32.174
    # Nfr = vm * vm / (g * d);
    lambdaL = vsl / vm
    lambdaG = vsg / vm
    NLV = 1.938 * vsl * np.power(rho_l / sigma_l, 0.25);       
    NGV = 1.938 * vsg * np.power(rho_l / sigma_l, 0.25);      
    ND = 120.872 * d * np.sqrt(rho_l / sigma_l);             
    NL = 0.15726 * mu_l * np.power(1 / (rho_l * np.power(sigma_l, 3)), 0.25);   

    #  No slip density
    rhoNS = rho_l * lambdaL + rho_g * (1 - lambdaL)

    flow_regime = FlowPattern(d, vsg, vsl, rho_l, sigma_l)

    HL = LiquidHoldUp(d, vsl, vsg, rho_l, mu_l, sigma_l, P)

    #  mixture viscosity (using noslip liquid holdup)
    muNS = mu_l * lambdaL + mu_g * (1 - lambdaL)
    muS = mu_l * HL + mu_g * (1 - HL)
    # slip density
    rhoS = rho_l * HL + rho_g * (1 - HL)

    Re = 1488 * rhoNS * vm * d / muS
    Bfac = 1.071 - 0.2218 * np.power(vm, 2) / d
    Bfac = max(0.13, Bfac)
    dpdzel = 0
    dpdzf = 0 
    dpdztpsi = 0 
    EK = 0
    frictionfactor = moody_friction_factor(e/d, Re)
    if lambdaG > Bfac:
        NLC = Correlationfunction_NLC(NL)
        aphi = NGV * np.power(NL, 0.380) / np.power(ND, 2.14)
        phi = Phi(d, vsg, vsl, rho_l, mu_l, sigma_l)
        aHLRatio = (NLV / np.power(NGV, 0.575)) * np.power(P / 14.7, 0.10) * (NLC / ND)
        HLphi = HLphiRatio(d, vsg, vsl, rho_l, mu_l, sigma_l, P)
        HL = HLphi * phi
        HL = max(HL, lambdaL)
        HL = max(0.00001, min(HL, 0.99999))
        
        dpdzel = rhoS * np.sin(angle) / 144
        dpdzf = frictionfactor * np.power(rhoNS, 2) * np.power(vm, 2) / (2 * g * 144 * rhoS * d)

        EK = vm * vsg * rhoNS / (P * g * 144)
        # EK = vm * vsg * rhoNS / (P * g * 144)
        # icrit = EK

        dpdztm = 0
        if EK > 0.95:
            dpdztm = (dpdzel + dpdzf)
        
        else:
            dpdztm = (dpdzel + dpdzf) / (1 - EK)
        
        dpdztpsi = dpdztm
    
    else:
        HLL = 0 
        slipdensity = 0
        # vs = 0.8
        HLL = Liquidholdup_Griffithwallis(vsg, vsl)
        HL = max(HLL, lambdaL)
        slipdensity = HL * rho_l + (1 - HL) * rho_g
        Rey = 1488 * rho_l * (vm / HL) * d / mu_l
        frictionfactor = moody_friction_factor(e/d, Rey)
        dpdzel = slipdensity * np.sin(angle) / 144
        dpdzf = frictionfactor * rho_l * np.power(vsl, 2) / (2 * 144 * g * d)
        
        dpdztpsi = dpdzf + dpdzel
        
    # dpdl_e = dpdzel
    # dpdl_f = dpdzf

    return dpdztpsi[0]


def LiquidHoldUp(d, vsl, vsg, rhoL, muL, sigmaL, P):
    hl_phi = HLphiRatio(d, vsg, vsl, rhoL, muL, sigmaL, P)
    phii = Phi(d, vsg, vsl, rhoL, muL, sigmaL)
    vm = vsg + vsl
    lambdaL = vsl / vm
    hld = hl_phi * phii
    HL = 0
    if hld < lambdaL:
        HL = lambdaL
    
    else:
        HL = hld
    
    return HL


def FlowPattern(d, vsg, vsl, rhoL, sigmaL):
    flowPat = ""
    NLV = 1.938 * vsl * np.power(rhoL / sigmaL, 0.25)
    NGV = 1.938 * vsg * np.power(rhoL / sigmaL, 0.25)
    ND = 120.872 * d * np.sqrt(rhoL / sigmaL)
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
    NGVBS = L1 + L2 * NLV   # Bubble/slug transition
    NGVSTR = 50 + 36 * NLV  # Slug/Transition Transition
    NGVTRM = 75 + 84 * np.power(NLV, 0.75)      # Transition/Mist transition

    if NGV < NGVBS:
        flowPat = "BUBBLE"
    
    elif (NGVBS < NGV) and (NGV < NGVSTR):
        flowPat = "SLUG"
    
    elif NGV > NGVTRM:
        flowPat = "MIST"
    
    elif NGV > NGVSTR and NGV < NGVTRM:
        flowPat = "TRANSITION"
    

    return flowPat


def Liquidholdup_Griffithwallis(vsg, vsl):
    vm = vsl + vsg
    lambdaL = vsl / vm
    vs = 0.8
    term1 = np.power(vs - vm, 2)
    term2 = 4 * vs * (vm - vsg)
    sum = term1 + term2
    HL = (vs - vm) + np.sqrt(sum) / (2 * vs)
    HL = max(HL, lambdaL)
    HL = max(0.00001, min(HL, 0.99999))

    return HL


def Correlationfunction_NLC(NL):
    NLC = 0
    if NL < 0.002:
        NLC = 0.02
    
    elif NL > 0.5:
        NLC = 0.01189207
    
    else:
        lognl = 0.0
        X = [ 0.002, 0.009, 0.020, 0.030, 0.04, 0.08, 0.100, 0.200, 0.300, 0.400, 0.500 ]
        nlc = [ 0.002, 0.0022133685, 0.003, 0.0032237, 0.0034641, 0.006, 
               0.0064807, 0.009, 0.01, 0.010905, 0.01189207 ]
        for i in range(len(X)):
            X[i] = np.log10(X[i])
            nlc[i] = np.log10(nlc[i])
        
        lognl = max(X[0], min(X[len(X) - 1], np.log10(NL)))
        NLC = np.power(10, np.interp(lognl, X, nlc))
    
    return NLC


def HLphiRatio(d, vsg, vsl, rhoL, muL, sigmaL, P):
    NLV = 1.938 * vsl * np.power(rhoL / sigmaL, 0.25); 
    NGV = 1.938 * vsg * np.power(rhoL / sigmaL, 0.25); 
    ND = 120.872 * d * np.sqrt(rhoL / sigmaL);       
    NL = 0.15726 * muL * np.power(1 / (rhoL * np.power(sigmaL, 3)), 0.25)  
    hlphi = 0
    nlc = Correlationfunction_NLC(NL)
    x = (NLV / np.power(NGV, 0.575)) * np.power(P / 14.7, 0.1) * (nlc / ND)
    if x <= 0.000002:
        hlphi = 0.05
    
    elif x > 0.005:
        hlphi = 0.98
    
    else:
        lognl = 0.0
        X = [ 0.000002, 0.000005, 0.00002, 0.00005, 0.00008, 0.0001, 
            0.000141421, 0.00024495, 0.0004, 0.0006, 0.001, 0.002, 0.005 ]
        hlfi = [ 0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.98 ]
        lognl = max(X[0], min(X[12], x))
        hlphi = np.interp(lognl, X, hlfi)
    
    return hlphi


def Phi(d, vsg, vsl, rhoL, muL, sigmaL):            
    NGV = 1.938 * vsg * np.power(rhoL / sigmaL, 0.25)
    ND = 120.872 * d * np.sqrt(rhoL / sigmaL)
    NL = 0.15726 * muL * np.power(1 / (rhoL * np.power(sigmaL, 3)), 0.25)
    x = NGV * np.power(NL, 0.380) / np.power(ND, 2.14)
    phi = 0
    if x < 0.01:
        phi = 1.0
    
    elif x > 0.09:
        phi = 1.82
    
    else:
        X = [ 0.01, 0.015, 0.0216667, 0.025, 0.030, 0.035, 
            0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090 ]
        fi = [ 1.0000, 1.0250, 1.1000, 1.2375, 1.4000, 1.5250, 1.6000,
              1.6500, 1.7000, 1.7500, 1.7800, 1.8000, 1.8200 ]
        lognl = max(X[0], min(X[len(X) - 1], x))
        phii = np.power(10, np.interp(lognl, X, fi))
        phi = phii
    
    return phi



if __name__ == "__main__":
    dpdl1 = hagedorn_brown_dpdl(1700, 90, 0.5, 0.00006, 5.88, 47.61, 0.016, 0.97, 8.41, 3.86, 3.97)
    print(f"{dpdl1 = }")







