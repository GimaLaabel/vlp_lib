import numpy as np
from utils import moody_friction_factor
from scipy.optimize import fsolve



def mukherjee_brill_dpdl(P: float, angle: float, d: float, e: float, rho_g: float, rho_l: float, mu_g: float, mu_l: float, sigma_l, vsg: float, vsl: float):
    angle = 0.01745329 * angle
    vm = vsg + vsl
    g = 32.174;    

    Nfr = vm * vm / (g * d)
    lambdaL = vsl / vm
    flowPat = FlowPattern(angle, d, vsg, vsl,rho_g, rho_l, mu_l, mu_g, sigma_l)            
        
    HL = LiquidHoldUp(angle, d, lambdaL, vsl, vsg, sigma_l, rho_l, mu_l, flowPat)

    # var srho = HL * rho_l + (1 - HL) * rho_g
    dpdztpsi = 0

    if flowPat =="BUBBLE" or flowPat=="SLUG":
        dpdztpsi = BubblePressureGradient(angle, d, P, vsg, vsl, rho_g, rho_l, mu_l, mu_g, sigma_l, e)
    
    elif flowPat=="STRATIFIED":
        dpdztpsi = StratifiedPressureGradient(angle, d, P, vsg, vsl, rho_g, rho_l, mu_l, mu_g, sigma_l, e)
    
    elif flowPat=="ANNULAR":
        dpdztpsi = AnnularPressureGradient(angle, d, P, vsg, vsl, rho_g, rho_l, mu_l, mu_g, sigma_l, e)
    
    return dpdztpsi[0]


def FlowDirection(angle: float) -> str:
    angle = 0.01745329 * angle
    if angle == 0:
        return "HORIZONTAL"
    
    if angle > 0:
        return "UPHILL" 
    else:
        return "DOWNHILL"

        

def LiquidHoldUp(angle, d, lambdaL, vsl, vsg, sigmaL, rhoL, muL, flowPat):
    # angle = 0.01745329 * angle;
    flowDir = FlowDirection(angle)

    NLV = 1.938 * vsl * np.power(rhoL / sigmaL, 0.25)      # Dimensionless liquid velocity number
    NGV = 1.938 * vsg * np.power(rhoL / sigmaL, 0.25)    # Dimensionless gas velocity number
    NL = 0.15726 * muL * np.power(1 / (rhoL * np.power(sigmaL, 3)), 0.25)

    C1 = CCoeff[flowPat][flowDir][0]
    C2 = CCoeff[flowPat][flowDir][1]
    C3 = CCoeff[flowPat][flowDir][2]
    C4 = CCoeff[flowPat][flowDir][3]
    C5 = CCoeff[flowPat][flowDir][4]
    C6 = CCoeff[flowPat][flowDir][5]

    HL = np.exp((C1 + C2 * np.sin(angle) + C3 * np.power(np.sin(angle), 2) + C4 * NL * NL) *
        (np.power(NGV, C5) / np.power(NLV, C6)))

    return HL


def FlowPattern(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL):
    # angle = 0.01745329 * angle;
    # Flow Pattern Map Coordinates

    flowPat = ""
    NLV = 1.938 * vsl * np.power(rhoL / sigmaL, 0.25)          # Dimensionless liquid velocity number
    NGV = 1.938 * vsg * np.power(rhoL / sigmaL, 0.25)          # Dimensionless gas velocity number
    NL = 0.15726 * muL * np.power(1 / (rhoL * np.power(sigmaL, 3)), 0.25)      # Dimensionless liquid viscosity number
    flowDir = FlowDirection(angle)

    # Uphill flow

    x = np.log10(NGV) + 0.940 + 0.074 * np.sin(angle) - 0.855 * np.power(np.sin(angle), 2) + 3.695 * NL
    NLVBS = np.power(10, x)
    NGVSM = np.power(10, (1.401 - 2.694 * NL + 0.521 * np.power(NLV, 0.329)))

    # Downflow and Horizontal flow

    y = 0.431 - 3.003 * NL - 1.138 * np.log10(NLV) * np.sin(angle) - 0.429 * np.power(np.log10(NLV), 2) * np.sin(angle) + 1.132 * np.sin(angle)

    NGVBS = np.power(10, y)
    z = 0.321 - 0.017 * NGV - 4.267 * np.sin(angle) - 2.972 * NL - 0.033 * np.power(np.log10(NGV), 2) - 3.925 * np.power(np.sin(angle), 2)

    NLVST = np.power(10, z)

    if flowDir =="UPHILL":
        if angle > 0 and NLV > NLVBS:
            flowPat = "BUBBLE"
        
        elif angle > 0 and not(NLV > NLVBS):
            flowPat = "SLUG"
        
        elif not(angle > 0) and NGV > NGVSM:
            flowPat = "ANNULAR"
        
    else:
        if (np.abs(angle) > 30 and not(NGV > NGVBS)) or ( not(np.abs(angle) > 30) and (NLV > NLVST) and not(NGV > NGVBS) ):
            flowPat = "BUBBLE"
        
        elif ((np.abs(angle) > 30) and (NGV > NGVBS) and (NLV > NLVST)) or ( not(np.abs(angle) > 30) and (NLV > NLVST) and (NGV > NGVBS)) :
            flowPat = "SLUG"
        
        elif (np.abs(angle) > 30) and (NGV > NGVBS) and not(NLV > NLVST) or ( not(np.abs(angle) > 30) and not(NLV > NLVST)):
            flowPat = "STRATIFIED"
        
    return flowPat
        

def BubblePressureGradient(angle, d, P, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL, e):
    # angle = 0.01745329 * angle;
     
    g = 32.174
    vm = vsg + vsl
    lambdaL = vsl / vm
    flowPat = FlowPattern(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL)
    HL = LiquidHoldUp(angle, d, lambdaL, vsl, vsg, sigmaL, rhoL, muL, flowPat)
    #  mixture viscosity (using noslip liquid holdup)
    muNS = muL * lambdaL + muG * (1 - lambdaL)
    #  No slip density
    rhoNS = rhoL * lambdaL + rhoG * (1 - lambdaL)
    #  slip density
    rhoS = rhoL * HL + rhoG * (1 - HL)
    #  reynolds no and
    Re = 1488 * rhoNS * vm * d / muNS

    frictionfactor = moody_friction_factor(e/d, Re)

    dpdzf = (frictionfactor * rhoS * vm * vm / (2 * d * g))
    dpdzel = rhoS * np.sin(angle)
    # double Ek;
    # Ek = vm * vsg * rhoNS / (P * 144)
    dpdztm = (dpdzf + dpdzel)
    dpdztpsi = dpdztm/144           #  (1 - Ek)
    return dpdztpsi


def AnnularPressureGradient(angle, d, P, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL, e):
    # angle = 0.01745329 * angle;
    
    g = 32.174
    vm = vsg + vsl
    #  Nfr = vm * vm / (g * d)
    lambdaL = vsl / vm
    flowPat = FlowPattern(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL)
    HL = LiquidHoldUp(angle, d, lambdaL, vsl, vsg, sigmaL, rhoL, muL, flowPat)
    HRR = lambdaL / HL
    #  mixture viscosity (using noslip liquid holdup)
    muNS = muL * lambdaL + muG * (1 - lambdaL)
    #  No slip density
    rhoNS = rhoL * lambdaL + rhoG * (1 - lambdaL)
    #  slip density
    rhoS = rhoL * HL + rhoG * (1 - HL)
    #  reynolds no and
    Re = rhoNS * vm * d / muNS
    mff = moody_friction_factor(e/d, Re)
    HR = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 10.00]
    fR = [1.00, 0.98, 1.20, 1.25, 1.30, 1.25, 1.00, 1.00]
    fRR = np.interp(HRR, HR, fR)
    ff = mff * fRR
    Ek = vm * vsg * rhoS / (P * 144)
    dpdzf = (ff * rhoNS * vm * vm  / (2 * d * g))
    dpdzel = rhoS * np.sin(angle)
    dpdztm = (dpdzf + dpdzel) / 144
    dpdztpsi = dpdztm
    return dpdztpsi


def StratifiedPressureGradient(angle, d, P, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL, e):
    # angle = 0.01745329 * angle;
    g = 32.174
    vm = vsg + vsl
    Nfr = vm * vm / (g * d)
    lambdaL = vsl / vm
    flowPat = FlowPattern(angle, d, vsg, vsl, rhoG, rhoL, muL, muG, sigmaL)
    HL = LiquidHoldUp(angle, d, lambdaL, vsl, vsg, sigmaL, rhoL, muL, flowPat)
    pipearea = np.pi * d * d / 4
    
    dpdztpsi = 0

    def funD(D):
        return (HL - (D - np.sin(D)) / (2 * np.pi))
                
    DD = fsolve(funD, 0.001)
    AL = pipearea * HL
    dhg = d * (2 * np.pi - (DD - np.sin(DD)) / (2 * np.pi - DD + 2 * np.sin(DD / 2)))
    dhl = d * (DD - np.sin(DD)) / (DD + 2 * np.sin(DD / 2))
    Pg = (1 - 0.5 * DD / np.pi) * P
    Pl = P - Pg
    vl = vsl / HL
    vg = vsg / (1 - HL)
    Nrel = Pl * vl * dhl / muL
    Nreg = Pg * vg * dhg / muG
    fl = moody_friction_factor(e/d, Nrel)
    fg = moody_friction_factor(e/d, Nreg)
    Twl = fl * Pl * vl * vl / (2 * g)
    Twg = fg * Pg * vg * vg / (2 * g)
    if d <= 0.2:
        dpdztpsi = -(Twg * Pg / (pipearea - AL)) - Pg * np.sin(angle)
    
    else:
        dpdztpsi = -(Twl * Pl + Twg * Pg) - (Pl * AL + Pg * (pipearea - AL)) * g * np.sin(angle)
    
    return dpdztpsi/(g * 144)


CCoeff = {    
    "BUBBLE": 
    {  
        "UPHILL": [-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657] ,
        "DOWNHILL": [-0.516644, 0.789805, 0.551627, 15.519214, 0.371771, 0.393952 ] 
    }, 

    "SLUG": 
    {
        "UPHILL": [-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657] ,
        "DOWNHILL": [-0.516644, 0.789805, 0.551627, 15.519214, 0.371771, 0.393952 ] 
    },                                                   
    
    "STRATIFIED":  
    {
        "UPHILL": [-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657],
        "DOWNHILL": [-1.330282, 4.808139, 4.171584, 56.262268, 0.079951, 0.504887] 
    },
    
    "ANNULAR": 
    {
        "UPHILL": [-0.380113, 0.129875, -0.119788, 2.343227, 0.475686, 0.288657] ,
        "DOWNHILL": [-0.516644, 0.789805, 0.551627, 15.519214, 0.371771, 0.393952 ] 
    }    
}




if __name__ == "__main__":

    dpdl1 = mukherjee_brill_dpdl(1700, 90, 0.5, 0.00006, 5.88, 47.61, 0.016, 0.97, 8.41, 3.86, 3.97)
    print(f"{dpdl1 = }")

