 
import torch

def chi(
    q,
    wavelength,
    C10,
    C30,
):
    """ """
    prefactor = 2*torch.pi / wavelength
    alpha = q*wavelength
    order_2 = alpha**2 / 2 * C10 
    order_4 = alpha**4 / 4 * C30
    
    return (order_2 + order_4) * prefactor

def complex_probe(
    q,
    wavelength,
    C10,
    C30,
    q_probe,
    reciprocal_sampling,
):
    """ """
    probe_array_fourier_0 = torch.sqrt(
    torch.clip(
            (q_probe - q)/reciprocal_sampling + 0.5,
            0,
            1,
        ),
    )
    probe_array_fourier_0 /= torch.sqrt(torch.sum(torch.abs(probe_array_fourier_0)**2))

    chi_ = chi(
        q,
        wavelength,
        C10,
        C30,
    )
    
    return probe_array_fourier_0 * torch.exp(-1j*chi_)

def ssb_ctf(
    complex_probe: torch.Tensor ,
    q,
    n,
    q_max,
):
 
    complex_probe_conj = complex_probe.conj()
    ssb_ctf = torch.zeros((n,n))
    for sx in range(n):
        for sy in range(n):
            if q[sx,sy] < q_max:
                shifted_probe_plus = torch.roll(complex_probe,(-sx,-sy),axis=(0,1))
                shifted_probe_minus = torch.roll(complex_probe,(sx,sy),axis=(0,1))
            
                gamma = complex_probe_conj * shifted_probe_minus - complex_probe * shifted_probe_plus.conj()
                ssb_ctf[sx,sy] = torch.abs(gamma).sum() / 2
            else:    
                ssb_ctf[sx,sy] = 0.0
    return ssb_ctf