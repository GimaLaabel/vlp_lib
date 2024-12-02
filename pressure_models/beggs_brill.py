from numpy import power, log, sin, sqrt, exp
from utils import friction_factor


def beggs_brill_dpdl(p: float, angle: float, d: float, e: float, rho_G: float, rho_L: float, mu_G: float, mu_L: float, sigma_L, vsG: float, vsL: float) -> float:
    angle_rad = 0.01745329 * angle
    vm = vsG + vsL
    g = 32.174
    Nfr = vm * vm/(g * d)
    lambdal = vsL/vm
    flowpattern = flow_pattern(d, vsG, vsL)
    holdup = 0

    # L1 = 316 * power(lambdal, 0.302)
    L2 = 0.000925 * power(lambdal, -2.468)
    L3 = 0.10 * power(lambdal, -1.452)
    # L4 = 0.5 * power(lambdal, -6.738)

    if flowpattern == "TRANSITION":
        hl_seg = liquid_holdup(angle, lambdal, Nfr, vsL, sigma_L, rho_L, "SEGREGATED")
        hl_int = liquid_holdup(angle, lambdal, Nfr, vsL, sigma_L, rho_L, "INTERMITTENT")
        A = (L3 - Nfr)/(L3 - L2)
        holdup = A * hl_seg + (1 - A) * hl_int
    else:
        holdup = liquid_holdup(angle, lambdal, Nfr, vsL, sigma_L, rho_L, flowpattern)
    
    hl = holdup
    # Payne et al holdup correction
    if angle > 0:
        holdup = 0.924 * holdup
    elif angle < 0:
        holdup = 0.685 * holdup
    
    mu_Ns = mu_L * lambdal + mu_G * (1 - lambdal)

    rho_Ns = rho_L * lambdal + rho_G * (1 - lambdal)
    rho_s = rho_L * holdup + rho_G * (1 - holdup)
    Re = 1488 * rho_Ns * vm *d /mu_Ns

    fn = friction_factor(d, e, Re)

    y = lambdal/(holdup * holdup)
    s = 0

    if 1 < y and y < 1.2:
        s = log(2.2 * y - 1.2)
    else: 
        denom = -0.0523 + 3.182 * log(y) - 0.8725 * power(log(y), 2) + 0.01853 * power(log(y), 4)
        nume = log(y)
        s = nume/denom
    
    f = fn * exp(s)

    Ek = vm * vsG * rho_Ns/(p * g * 144)

    dpdl_f = f * rho_Ns * vm * vm /(2 * d * g)
    dpdl_e = rho_s * sin(angle_rad)

    dpdl = (dpdl_f + dpdl_e)/144
    dpdl = dpdl/(1 - Ek)
    
    return {"dpdl": dpdl[0], "liquidholdup": holdup, "frictionfactor": f, "flowpattern": flowpattern, "lambdal": lambdal}


def liquid_holdup(angle: float, lamdal: float, Nfr: float, vsl: float, sigmaL: float, rhoL: float, flow_pat: str) -> float:
    g = 32.174
    a = Hcoeff[flow_pat][0]    
    b = Hcoeff[flow_pat][1]
    c = Hcoeff[flow_pat][2]
    angle_rad = angle * 0.01745329

    hlo = a * power(lamdal, b)/power(Nfr, c)
    flow_dir = flow_direction(angle)

    e = Ccoeff[flow_pat][flow_dir][0]
    f = Ccoeff[flow_pat][flow_dir][1]
    gg = Ccoeff[flow_pat][flow_dir][2]
    h = Ccoeff[flow_pat][flow_dir][3]

    Nlv = 1.938 * vsl * power(rhoL/(sigmaL), 0.25)
    # C = (1.0 - lamdal) * log(e * np.power(lamdal, f) * np.power(Nlv, gg) * np.power(Nfr, h))
    C = (1.0 - lamdal) * log(e * power(lamdal, f) * power(Nlv, gg) * power(Nfr, h))
    C = max(0, C)

    psi = 1.0 + C * (sin(1.8 * angle_rad) - 0.333 * power(sin(1.8 * angle_rad), 3))
    if flow_pat == "DISTRIBUTED" and flow_dir == "UPHILL":
        C = 0.0
        psi = 1.0
    Hl = hlo * psi 
    
    return Hl 

  
def flow_direction(angle: float) -> str:
    # angle = 0.01745329 * angle
    if angle == 0:
        return "HORIZONTAL"
    if angle > 0:
        return "UPHILL"
    else:
        return "DOWNHILL"

def flow_pattern(d: float, vsg: float, vsl: float) -> str:
    g = 32.174
    vm = vsl + vsg
    Nfr = vm * vm/(g * d)
    lambdal = vsl/vm
    L1 = 316 * power(lambdal, 0.302)
    L2 = 0.000925 * power(lambdal, -2.468)
    L3 = 0.1 * power(lambdal, -1.452)
    L4 = 0.5 * power(lambdal, -6.738)
    flow_pat = "DISTRIBUTED"
    
    if (lambdal < 0.4 and Nfr >= L1) or (lambdal >= 0.4 and Nfr > L4):
        flow_pat = "DISTRIBUTED"
    elif (lambdal < 0.01 and Nfr < L1) or (lambdal >= 0.01 and Nfr < L2):
        flow_pat = "SEGREGATED"
    elif (lambdal >= 0.01) and (L2 <= Nfr and Nfr <= L3):
        flow_pat = "TRANSITION"
    elif ((0.01 <= lambdal and lambdal <0.4) and (L3 < Nfr and Nfr <= L1)) or (lambdal >= 0.4 and (L3 < Nfr and Nfr <= L4)):
        flow_pat = "INTERMITTENT"
    
    return flow_pat


Hcoeff = {
    "SEGREGATED": [0.9800, 0.4846, 0.0868],
    "INTERMITTENT": [0.8450, 0.5351, 0.0173],
    "DISTRIBUTED": [1.065, 0.5824, 0.0609]
} 

Ccoeff = {
    "SEGREGATED": {
        "UPHILL": [0.011, -3.7680, 3.5390, -1.6140],
        "DOWNHILL": [4.700, -0.3692, 0.1244, -0.5056]
        },
    "INTERMITTENT": {
        "UPHILL": [2.960, 0.3050, -0.4473, 0.0978], 
        "DOWNHILL": [4.700, -0.3692, 0.1244, -0.5056]
    }, 
    "DISTRIBUTED": {
            "UPHILL": [1.0, 0, 0, 0],
            "DOWNHILL": [4.700, -0.3692, 0.1244, -0.5056]
    }    
}
 


if __name__ == "__main__":
    # print(f"{np.sin(1.57) = }")
    # dire = flow_direction(0)
    # print(dire)

    # print(Hcoeff["SEGREGATED"][1])

    # ed = [0.000144, 0.00012, 0.00012] 
    # re = [3.688e5, 1.45e5, 2.86e5]
    # ff = []
    # for i in range(len(re)):
    #     res = moody_friction_factor(ed[i], re[i])
    #     ff.append(res)
    #     print(f"Friction factor for {ed[i]}, {re[i]}  = {res}")

    # print(f"{Hcoeff["DISTRIBUTED"][0] = }")
    # print(f"{Ccoeff['SEGREGATED']['UPHILL'][2] = }")

    dpdl1 = beggs_brill_dpdl(1700, 90, 0.5, 0.00006, 5.88, 47.61, 0.016, 0.97, 8.41, 3.86, 3.97)
    print(f"{dpdl1 = }")

    # dpdl2 = beggs_brill_dpdl(720, 90, 0.249, 0.00006, 2.84, 56.6, 0.018, 18.0, 8.41, 4.09, 2.65)

    # print(f"{dpdl2 = }")

